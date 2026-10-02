"""Paired 20 s evaluation for the historical-A6 HoST adaptation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.host_recovery.a6_runtime import (
  SCENE_NAMES,
  _a6_ground_force,
  _a6_height,
  _a6_upright,
  a6_env_cfg,
  a6_reset_counts,
)
from src.tasks.host_recovery.config.g1.rl_cfg import (
  unitree_g1_host_standup_ppo_runner_cfg,
)
from src.tasks.host_recovery.mdp.a6_geometry import DIRECTIONS
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


VALIDATION_BANKS = {
  "outputs/multiterrain_bank/validation.npz": (
    "d0c4755474df9626a20167843eb453d00e76420771d29f34cd35d5a3a8d29b45"
  ),
  "datasets/reset_banks/natural_curriculum_v1/validation.npz": (
    "7c0f10a217f05673cf987e7094d6be449228f5546756f11d1424f428bff0dfaf"
  ),
  "datasets/reset_banks/procedural_low_v1/validation.npz": (
    "eb0d2d2ab1131386c21f70d10dc5be36444728bb0581c452fdc3bb3551bb6d31"
  ),
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _fraction(value: torch.Tensor) -> float:
  return value.float().mean().item()


def _mean(value: torch.Tensor) -> float:
  return value.float().mean().item()


def _p95(value: torch.Tensor) -> float:
  return torch.quantile(value.float(), 0.95).item()


def _set_action_scale(env, value: float) -> None:
  term = env.action_manager.get_term("joint_pos")
  if isinstance(term.cfg.scale, dict):
    term.cfg.scale = {key: value for key in term.cfg.scale}
  else:
    term.cfg.scale = value
  if isinstance(term._scale, torch.Tensor):
    term._scale[:] = value
  else:
    term._scale = value
  env._host_action_rescale = torch.full(
    (env.num_envs, 1), value, device=env.device
  )


def _stability_components(env) -> dict[str, torch.Tensor]:
  """Return every predicate in the frozen historical stability checker."""
  robot = env.scene["robot"]
  height = _a6_height(env)
  upright = _a6_upright(env)
  feet_pos = robot.data.body_link_pos_w[:, env._a6_feet_bodies, :]
  width = (feet_pos[:, 0, :2] - feet_pos[:, 1, :2]).norm(dim=-1)
  foot_speed = robot.data.body_link_lin_vel_w[
    :, env._a6_feet_bodies, :
  ].norm(dim=-1).amax(-1)
  foot_load = _a6_ground_force(env, "quality_feet")[..., 2].abs().amin(-1)
  other_load = _a6_ground_force(env, "quality_other")[..., 2].abs().sum(-1)
  base_speed = robot.data.root_link_lin_vel_w.norm(dim=-1)
  angular_speed = robot.data.root_link_ang_vel_w.norm(dim=-1)
  joint_rms = robot.data.joint_vel.square().mean(-1).sqrt()
  knee = robot.data.joint_pos[:, env._a6_knee_joints].abs().amax(-1)
  return {
    "height": height >= 1.15,
    "upright": upright >= 0.93,
    "knee": knee < 0.8,
    "base_speed": base_speed < 0.15,
    "angular_speed": angular_speed < 0.3,
    "joint_speed": joint_rms < 0.5,
    "foot_speed": foot_speed < 0.1,
    "foot_load": foot_load > 20.0,
    "other_contact": other_load < 20.0,
    "foot_width_low": width >= 0.12,
    "foot_width_high": width <= 0.45,
  }


def _stable(components: dict[str, torch.Tensor]) -> torch.Tensor:
  return torch.stack(tuple(components.values()), dim=-1).all(-1)


def _summarize(mask: torch.Tensor, arrays: dict[str, torch.Tensor]) -> dict:
  plate = arrays["plate"][mask]
  escaped = arrays["escaped"][mask]
  sr1 = arrays["sr1"][mask]
  sr10 = arrays["sr10"][mask]
  result = {
    "n": int(mask.sum()),
    "clear_rate": _fraction(escaped[plate]) if plate.any() else None,
    "sr1": _fraction(sr1),
    "sr10": _fraction(sr10),
    "core_sr1": _fraction(arrays["core_sr1"][mask]),
    "core_sr10": _fraction(arrays["core_sr10"][mask]),
    "early_termination": _fraction(arrays["early_done"][mask]),
    "invalid_plate": _fraction(arrays["invalid"][mask]),
    "refall_after_upright": _fraction(arrays["refall"][mask]),
    "best_stable_hold_seconds_mean": _mean(arrays["best_hold"][mask]),
    "first_escape_seconds_mean_success": (
      _mean(arrays["first_escape"][mask][escaped]) if escaped.any() else None
    ),
    "first_sr1_seconds_mean_success": (
      _mean(arrays["first_sr1"][mask][sr1]) if sr1.any() else None
    ),
    "tau_peak_p95": _p95(arrays["tau_peak"][mask]),
    "joint_speed_peak_p95": _p95(arrays["speed_peak"][mask]),
    "joint_power_peak_p95": _p95(arrays["power_peak"][mask]),
    "high_load_time_p95": _p95(arrays["high_load_time"][mask]),
    "longest_high_load_p95": _p95(arrays["longest_high_load"][mask]),
    "stability_component_true_fraction": {
      name.removeprefix("stability_fraction_"): _mean(value[mask])
      for name, value in arrays.items()
      if name.startswith("stability_fraction_")
    },
  }
  return result


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--group", choices=("G-", "G+"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--seed", type=int, default=20261021)
  parser.add_argument(
    "--condition", choices=("nominal", "randomized"), default="nominal"
  )
  args = parser.parse_args()
  if args.num_envs < 32 or args.num_envs % 32:
    raise ValueError("num-envs must be >=32 and divisible by 32")

  bank_hashes = {name: _sha256(Path(name)) for name in VALIDATION_BANKS}
  if bank_hashes != VALIDATION_BANKS:
    raise RuntimeError(f"validation bank hash mismatch: {bank_hashes}")

  cfg = a6_env_cfg(
    play=True,
    seed=args.seed,
    dynamics=args.condition == "randomized",
    group=args.group,
    bank_split="validation",
  )
  cfg.scene.num_envs = args.num_envs
  cfg.episode_length_s = args.steps * 0.02
  # Keep the first episode in-place so the exact invalid-plate flag can be
  # captured before an automatic reset clears it. Invalid worlds are censored
  # locally from all later evaluation statistics.
  cfg.terminations.pop("invalid_plate", None)
  if args.condition == "nominal":
    for event in ("foot_friction", "encoder_bias", "base_com", "push_robot"):
      cfg.events.pop(event, None)
  else:
    cfg.observations["actor"].terms["host"].params["add_noise"] = True
    cfg.events["push_robot"] = EventTermCfg(
      func=push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        }
      },
    )

  agent = unitree_g1_host_standup_ppo_runner_cfg()
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
  if args.condition == "randomized":
    # The wrapper's first reset sees the fully expanded training ranges:
    # +/-20% grouped dynamics and a 4--12 kg plate.
    env._a6_adaptation_start_step = -100_000
  wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  try:
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), device="cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=False, map_location="cuda:0")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_scale = checkpoint.get("infos", {}).get("env_state", {}).get(
      "host_action_rescale"
    )
    if saved_scale is None:
      raise RuntimeError("checkpoint has no host_action_rescale state")
    action_scale = float(saved_scale.float().mean())
    _set_action_scale(env, action_scale)

    policy = runner.get_inference_policy(device="cuda:0")
    obs = wrapper.get_observations()
    robot = env.scene["robot"]
    initial_qpos = env.sim.data.qpos.detach().cpu().contiguous().numpy()
    initial_qpos_sha = hashlib.sha256(initial_qpos.tobytes()).hexdigest()
    n = env.num_envs
    device = env.device
    alive = torch.ones(n, dtype=torch.bool, device=device)
    early_done = torch.zeros_like(alive)
    ever_invalid = torch.zeros_like(alive)
    ever_escaped = torch.zeros_like(alive)
    ever_upright = torch.zeros_like(alive)
    refall = torch.zeros_like(alive)
    fall_time = torch.zeros(n, device=device)
    hold = torch.zeros_like(fall_time)
    best_hold = torch.zeros_like(fall_time)
    core_hold = torch.zeros_like(fall_time)
    best_core_hold = torch.zeros_like(fall_time)
    first_escape = torch.full_like(fall_time, -1.0)
    first_sr1 = torch.full_like(fall_time, -1.0)
    first_sr10 = torch.full_like(fall_time, -1.0)

    joints = robot.data.joint_pos.shape[-1]
    env._a6_eval_active = alive.clone()
    env._a6_eval_tau_peak = torch.zeros(n, joints, device=device)
    env._a6_eval_speed_peak = torch.zeros(n, joints, device=device)
    env._a6_eval_power_peak = torch.zeros(n, joints, device=device)
    env._a6_eval_high_load_time = torch.zeros(n, joints, device=device)
    env._a6_eval_longest_high_load = torch.zeros(n, joints, device=device)

    plate = env._a6_scene > 0
    component_time: dict[str, torch.Tensor] = {}
    evaluated_time = torch.zeros(n, device=device)
    with torch.no_grad():
      for step in range(args.steps):
        actions = policy(obs["actor"])
        obs, _, dones, _ = wrapper.step(actions)
        is_early = dones.bool() & (step + 1 < args.steps)
        early_done |= is_early
        invalid_now = env._a6_invalid & alive
        ever_invalid |= invalid_now
        alive &= ~(is_early | invalid_now)
        env._a6_eval_active.copy_(alive)

        escaped_now = env._a6_escaped & alive
        newly_escaped = escaped_now & ~ever_escaped
        first_escape[newly_escaped] = (step + 1) * env.step_dt
        ever_escaped |= escaped_now

        components = _stability_components(env)
        if not component_time:
          component_time = {
            name: torch.zeros(n, device=device) for name in components
          }
        evaluated_time += alive * env.step_dt
        for name, value in components.items():
          component_time[name] += value * alive * env.step_dt

        eligibility = alive & (~plate | escaped_now)
        stable = _stable(components) & eligibility
        hold = torch.where(stable, hold + env.step_dt, torch.zeros_like(hold))
        best_hold = torch.maximum(best_hold, hold)
        core_stable = components["height"] & components["upright"] & eligibility
        core_hold = torch.where(
          core_stable, core_hold + env.step_dt, torch.zeros_like(core_hold)
        )
        best_core_hold = torch.maximum(best_core_hold, core_hold)
        new_sr1 = (hold >= 1.0 - 1e-4) & (first_sr1 < 0)
        new_sr10 = (hold >= 10.0 - 1e-4) & (first_sr10 < 0)
        first_sr1[new_sr1] = (step + 1) * env.step_dt - 1.0
        first_sr10[new_sr10] = (step + 1) * env.step_dt - 10.0

        upright_now = (_a6_height(env) >= 1.15) & (_a6_upright(env) >= 0.93)
        ever_upright |= upright_now & alive
        fallen = (_a6_height(env) < 0.65) | (_a6_upright(env) < 0.5)
        fall_time = torch.where(
          fallen & alive, fall_time + env.step_dt, torch.zeros_like(fall_time)
        )
        refall |= ever_upright & (fall_time >= 0.2)

    sr1 = best_hold >= 1.0 - 1e-4
    sr10 = best_hold >= 10.0 - 1e-4
    core_sr1 = best_core_hold >= 1.0 - 1e-4
    core_sr10 = best_core_hold >= 10.0 - 1e-4
    trial_peak = lambda value: value.amax(-1)  # noqa: E731
    arrays = {
      "plate": plate,
      "escaped": ever_escaped,
      "sr1": sr1,
      "sr10": sr10,
      "core_sr1": core_sr1,
      "core_sr10": core_sr10,
      "early_done": early_done,
      "invalid": ever_invalid,
      "refall": refall,
      "best_hold": best_hold,
      "first_escape": first_escape,
      "first_sr1": first_sr1,
      "tau_peak": trial_peak(env._a6_eval_tau_peak),
      "speed_peak": trial_peak(env._a6_eval_speed_peak),
      "power_peak": trial_peak(env._a6_eval_power_peak),
      "high_load_time": trial_peak(env._a6_eval_high_load_time),
      "longest_high_load": trial_peak(env._a6_eval_longest_high_load),
      **{
        f"stability_fraction_{name}": value / evaluated_time.clamp_min(env.step_dt)
        for name, value in component_time.items()
      },
    }

    masks = {name: env._a6_scene == idx for idx, name in enumerate(SCENE_NAMES)}
    for scene_idx, scene_name in enumerate(SCENE_NAMES):
      for direction_idx, direction_name in enumerate(DIRECTIONS):
        masks[f"{scene_name}/{direction_name}"] = (
          (env._a6_scene == scene_idx) & (env._a6_direction == direction_idx)
        )
    summary = {
      name: _summarize(mask, arrays) for name, mask in masks.items() if mask.any()
    }
    scene_rows = [summary[name] for name in SCENE_NAMES]
    macro = {
      metric: float(np.mean([row[metric] for row in scene_rows]))
      for metric in ("sr1", "sr10", "core_sr1", "core_sr10")
    }
    macro["plate_clear_rate"] = float(
      np.mean([summary[name]["clear_rate"] for name in SCENE_NAMES[1:]])
    )

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": _sha256(args.checkpoint),
      "group": args.group,
      "condition": args.condition,
      "split": "historical_validation",
      "not_independent_final_test": True,
      "seed": args.seed,
      "num_envs": args.num_envs,
      "steps": args.steps,
      "seconds": args.steps * env.step_dt,
      "action_scale": action_scale,
      "bank_sha256": bank_hashes,
      "initial_qpos_sha256": initial_qpos_sha,
      "reset_counts": a6_reset_counts(env),
      "evaluation_condition": {
        "plate_mass_kg": 6.0 if args.condition == "nominal" else [4.0, 12.0],
        "a6_dynamics_randomization": args.condition == "randomized",
        "grouped_dynamics_range": (
          None if args.condition == "nominal" else [-0.2, 0.2]
        ),
        "reset_randomization_events": (
          []
          if args.condition == "nominal"
          else ["foot_friction", "encoder_bias", "base_com"]
        ),
        "push_interval_seconds": (
          None if args.condition == "nominal" else [1.0, 3.0]
        ),
        "actor_observation_noise": args.condition == "randomized",
      },
      "stability_definition": {
        "height_min_m": 1.15,
        "upright_min": 0.93,
        "knee_abs_max_rad": 0.8,
        "base_speed_max_mps": 0.15,
        "angular_speed_max_radps": 0.3,
        "joint_speed_rms_max_radps": 0.5,
        "foot_speed_max_mps": 0.1,
        "both_feet_load_min_n": 20.0,
        "other_ground_load_max_n": 20.0,
        "foot_width_m": [0.12, 0.45],
      },
      "macro": macro,
      "summary": summary,
    }
    (args.output / "summary.json").write_text(
      json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
      args.output / "per_trial.npz",
      qpos=initial_qpos,
      scene=env._a6_scene.cpu().numpy(),
      direction=env._a6_direction.cpu().numpy(),
      stratum=env._a6_stratum.cpu().numpy(),
      source=env._a6_source.cpu().numpy(),
      **{name: value.cpu().numpy() for name, value in arrays.items()},
    )
    print("A6_HOST_EVALUATION_COMPLETE", json.dumps(macro), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()

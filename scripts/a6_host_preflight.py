"""Executable runtime preflight for the historical A6 HoST adapter.

This is not a training launch script. It instantiates the three-scene runtime
on GPU and verifies the executable reset/dynamics layers:

- bank SHAs and scene/direction/stratum/source allocations;
- 29-joint robot plus guided/free plate entities and contact sensors;
- zero initial velocities;
- grouped mass/inertia/Kp/Kd randomisation and 0--10 ms command delay;
- real board placement for scene 1/2 and parked boards for flat scenes;
- partial-reset isolation of untouched worlds;
- a short finite rollout under the native HoST observation/action shapes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from src.tasks.host_recovery.a6_runtime import (
  A6_DYNAMICS_WARMUP_UPDATES,
  A6_STEPS_PER_UPDATE,
  A6_ROOT,
  LOW_BANK_PATH,
  NATURAL_BANK_PATH,
  PROCEDURAL_BANK_PATH,
  a6_audit_dynamics,
  a6_env_cfg,
  a6_reset_counts,
)

EXPECTED_SHA256 = {
  str(LOW_BANK_PATH):
    "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a",
  str(NATURAL_BANK_PATH):
    "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9",
  str(PROCEDURAL_BANK_PATH):
    "e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _bincount(tensor: torch.Tensor, minlength: int) -> list[int]:
  return torch.bincount(tensor, minlength=minlength).tolist()


def _direction_from_gravity(gravity: torch.Tensor) -> torch.Tensor:
  """Historical four-posture direction inferred from projected gravity."""
  x_dominant = gravity[:, 0].abs() >= gravity[:, 1].abs()
  return torch.where(
    x_dominant,
    torch.where(gravity[:, 0] > 0, 1, 0),
    torch.where(gravity[:, 1] > 0, 2, 3),
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=4)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--root", type=Path, default=Path.cwd())
  parser.add_argument("--log-dir", type=Path, default=None)
  args = parser.parse_args()

  root = args.root.resolve()
  if args.log_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.log_dir = root / "logs" / f"a6_host_preflight_{stamp}"
  args.log_dir.mkdir(parents=True, exist_ok=False)

  bank_sha = {}
  for relative, expected in EXPECTED_SHA256.items():
    path = root / relative
    if not path.is_file():
      raise FileNotFoundError(f"missing A6 bank: {path}")
    actual = _sha256(path)
    if actual != expected:
      raise ValueError(
        f"SHA256 mismatch for {relative}: expected={expected} actual={actual}"
      )
    bank_sha[relative] = actual

  cfg = a6_env_cfg(play=True, seed=args.seed, dynamics=True)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    robot = env.scene["robot"]
    guided = env.scene["escape_obstacle"]
    free = env.scene["free_obstacle"]
    assert len(robot.joint_names) == 29, robot.joint_names
    assert "escape_plate_slide" in guided.joint_names, guided.joint_names
    assert free.indexing.free_joint_q_adr.numel() == 7

    obs, _ = env.reset()
    assert tuple(obs["actor"].shape) == (args.num_envs, 564), obs["actor"].shape
    assert tuple(obs["critic"].shape) == (args.num_envs, 564), obs["critic"].shape
    assert env.sim.data.qvel.abs().max() < 1e-6

    counts = a6_reset_counts(env)
    n = args.num_envs
    expected_scene = [n // 2, n // 4, n // 4]
    expected_direction = [n // 4] * 4
    assert counts["scene"] == expected_scene, (counts["scene"], expected_scene)
    assert counts["direction"] == expected_direction, (
      counts["direction"], expected_direction
    )
    for actual, expected in zip(counts["stratum"], (0.70 * n, 0.20 * n, 0.10 * n)):
      assert abs(actual - expected) <= max(3, round(0.05 * n)), (
        counts["stratum"], expected
      )
    low_count = counts["stratum"][0]
    source_fraction = counts["source"][1] / max(low_count, 1)
    assert source_fraction <= 0.31, counts
    if n >= 64:
      assert source_fraction >= 0.15, counts

    dynamics_audit = a6_audit_dynamics(env)
    assert dynamics_audit["body_mass_inertia_verified"]
    assert dynamics_audit["actuator_gain_verified"]
    assert dynamics_audit["effort_caps_unchanged"]
    assert dynamics_audit["delay_0_to_10ms_verified"]
    assert dynamics_audit["plate_mass_verified"]

    low_ids = torch.nonzero(env._a6_stratum == 0, as_tuple=False).flatten()
    inferred = _direction_from_gravity(robot.data.projected_gravity_b)
    assert torch.equal(
      inferred[low_ids], env._a6_direction[low_ids]
    ), _bincount(inferred[low_ids], 4)
    assert (
      robot.data.projected_gravity_b[low_ids, :2].norm(dim=-1) > 0.5
    ).all()

    # Board placement: scene 1/2 are directly under the robot; flat boards are
    # parked far away. Geom positions are local to each entity's geom list.
    guided_geom, _ = guided.find_geoms("escape_plate_geom")
    free_geom, _ = free.find_geoms("plate_geom")
    guided_xy = guided.data.geom_pos_w[:, guided_geom[0], :2]
    free_xy = free.data.geom_pos_w[:, free_geom[0], :2]
    robot_xy = robot.data.root_link_pos_w[:, :2]

    for scene_id, plate_xy in ((1, guided_xy), (2, free_xy)):
      ids = torch.nonzero(env._a6_scene == scene_id, as_tuple=False).flatten()
      assert torch.norm(plate_xy[ids] - robot_xy[ids], dim=-1).max() < 0.05
    flat_ids = torch.nonzero(env._a6_scene == 0, as_tuple=False).flatten()
    assert torch.norm(guided_xy[flat_ids] - robot_xy[flat_ids], dim=-1).min() > 10.0
    assert torch.norm(free_xy[flat_ids] - robot_xy[flat_ids], dim=-1).min() > 10.0

    for sensor_name in ("guided_contact", "free_contact", "path_hands"):
      sensor = env.scene[sensor_name]
      assert hasattr(sensor, "data"), sensor_name

    # Partial reset must touch only the requested worlds.
    ids = torch.tensor(
      [0, n // 4, n // 2, 3 * n // 4], device=env.device, dtype=torch.long
    )
    mask = torch.ones(n, dtype=torch.bool, device=env.device)
    mask[ids] = False
    before = {
      "qpos": env.sim.data.qpos[mask].clone(),
      "qvel": env.sim.data.qvel[mask].clone(),
      "mocap_pos": env.sim.data.mocap_pos[mask].clone(),
      "mocap_quat": env.sim.data.mocap_quat[mask].clone(),
      "best_score": env._a6_best_score[mask].clone(),
      "stratum": env._a6_stratum[mask].clone(),
      "mass_factors": env._a6_mass_factors[mask].clone(),
      "gain_factors": env._a6_gain_factors[mask].clone(),
      "lag": env._a6_lag[mask].clone(),
      "nominal": env._a6_nominal[mask].clone(),
      "body_mass": env.sim.model.body_mass[mask].clone(),
      "body_inertia": env.sim.model.body_inertia[mask].clone(),
    }
    env._reset_idx(ids)
    env.sim.forward()
    assert torch.equal(before["qpos"], env.sim.data.qpos[mask])
    assert torch.equal(before["qvel"], env.sim.data.qvel[mask])
    assert torch.equal(before["mocap_pos"], env.sim.data.mocap_pos[mask])
    assert torch.equal(before["mocap_quat"], env.sim.data.mocap_quat[mask])
    assert torch.equal(before["best_score"], env._a6_best_score[mask])
    assert torch.equal(before["stratum"], env._a6_stratum[mask])
    assert torch.equal(before["mass_factors"], env._a6_mass_factors[mask])
    assert torch.equal(before["gain_factors"], env._a6_gain_factors[mask])
    assert torch.equal(before["lag"], env._a6_lag[mask])
    assert torch.equal(before["nominal"], env._a6_nominal[mask])
    assert torch.equal(before["body_mass"], env.sim.model.body_mass[mask])
    assert torch.equal(before["body_inertia"], env.sim.model.body_inertia[mask])
    assert env.sim.data.qvel[ids].abs().max() < 1e-6

    # Width schedule: at 2000 PPO updates the support is +/-20%, not the
    # initial +/-10%.  Audit column 3 stays nominal in both regimes.
    saved_counter = env.common_step_counter
    env.common_step_counter = A6_STEPS_PER_UPDATE * A6_DYNAMICS_WARMUP_UPDATES
    obs, _ = env.reset()
    non_nominal = ~env._a6_nominal
    assert non_nominal.any()
    assert env._a6_mass_factors[non_nominal, :3].min() >= 0.8
    assert env._a6_mass_factors[non_nominal, :3].max() <= 1.2
    assert env._a6_gain_factors[non_nominal].min() >= 0.8
    assert env._a6_gain_factors[non_nominal].max() <= 1.2
    assert torch.equal(
      env._a6_mass_factors[:, 3],
      torch.ones(n, device=env.device),
    )
    assert env._a6_plate_mass.min() >= 4.0
    assert env._a6_plate_mass.max() <= 12.0
    env.common_step_counter = saved_counter

    # Restore a coherent full state, then run a short finite rollout through the
    # native wrapper/observation shapes.
    obs, _ = env.reset()
    wrapper = RslRlVecEnvWrapper(env)
    obs = wrapper.get_observations()
    actions = torch.zeros((n, 29), device=env.device)
    for _ in range(args.steps):
      obs, reward, _, _ = wrapper.step(actions)
      assert all(torch.isfinite(value).all() for value in obs.values())
      assert torch.isfinite(reward).all()
      assert torch.isfinite(env.sim.data.qpos).all()
      assert torch.isfinite(env.sim.data.qvel).all()

    report = {
      "status": "A6_HOST_DYNAMICS_PREFLIGHT_PASS",
      "num_envs": n,
      "seed": args.seed,
      "device": args.device,
      "steps": args.steps,
      "robot_joints": len(robot.joint_names),
      "actor_obs_dim": tuple(obs["actor"].shape),
      "critic_obs_dim": tuple(obs["critic"].shape),
      "num_actions": 29,
      "bank_sha256": bank_sha,
      "reset_counts": counts,
      "scene_names": ["flat", "vertical_plate", "free_plate"],
      "stratum_names": ["low", "middle", "near_standing"],
      "entities": list(env.scene._entities.keys()),
      "sensors": ["guided_contact", "free_contact", "path_hands"],
      "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
      ).strip(),
      "dynamics_audit": dynamics_audit,
      "dynamics_width_verified": True,
      "note": "runtime entities, exact reset, and grouped dynamics are executable; reward layer remains pending",
    }
    (args.log_dir / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print("A6_HOST_DYNAMICS_PREFLIGHT_PASS", json.dumps(report, sort_keys=True), flush=True)
    return 0
  finally:
    env.close()


if __name__ == "__main__":
  raise SystemExit(main())

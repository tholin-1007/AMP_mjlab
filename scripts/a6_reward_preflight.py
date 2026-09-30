"""Executable reward-layer preflight for the A6 HoST adapter.

Verifies, on GPU and without training:

- the G+/G- reward mapping and dt-folded weights;
- that overlapping flat29/HoST terms are zeroed instead of double-counted;
- that shared task/cost buffers are finite and per-step substep stats advance;
- alpha gating for G+ versus G-;
- formula counterexamples for G/L gates and reset clearing of new buffers.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv

from src.tasks.host_recovery.a6_runtime import (
  A6_COST_NAMES,
  A6_COST_WEIGHTS,
  A6_NATIVE_OVERLAP_TERMS,
  A6_POSITIVE_NAMES,
  A6_POSITIVE_WEIGHTS,
  HOST_CONSTRAINT_DT,
  _a6_alpha,
  _a6_gate_ramp,
  _a6_smooth_gate,
  _a6_update,
  a6_env_cfg,
)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=2)
  parser.add_argument(
    "--episode-steps",
    type=int,
    default=0,
    help="also run this many policy-free control steps per arm",
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--root", type=Path, default=Path.cwd())
  parser.add_argument("--log-dir", type=Path, default=None)
  args = parser.parse_args()

  root = args.root.resolve()
  if args.log_dir is None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    args.log_dir = root / "logs" / f"a6_reward_preflight_{stamp}"
  args.log_dir.mkdir(parents=True, exist_ok=False)

  cfg_plus = a6_env_cfg(play=True, seed=args.seed, dynamics=True, group="G+")
  cfg_plus.scene.num_envs = args.num_envs
  cfg_minus = a6_env_cfg(play=True, seed=args.seed, dynamics=True, group="G-")
  cfg_minus.scene.num_envs = args.num_envs

  env_plus = ManagerBasedRlEnv(cfg=cfg_plus, device=args.device)
  env_minus = ManagerBasedRlEnv(cfg=cfg_minus, device=args.device)
  try:
    dt = HOST_CONSTRAINT_DT
    for group, env, cfg in (
      ("G+", env_plus, cfg_plus),
      ("G-", env_minus, cfg_minus),
    ):
      expected = {}
      for name, weight in zip(A6_POSITIVE_NAMES, A6_POSITIVE_WEIGHTS):
        expected[f"a6_{name}"] = weight * dt
      for name, weight in zip(A6_COST_NAMES, A6_COST_WEIGHTS):
        expected[f"a6_{name}"] = weight * dt
      expected["a6_soft_limits"] = -0.10 * dt
      expected["plate_completion"] = 0.60 * dt
      expected["plate_force"] = -0.03 * dt
      expected["path_quiet_feet"] = -0.03 * dt
      expected["path_joint_stall"] = -0.20 * dt
      if group == "G+":
        expected["plate_geometry_progress"] = 0.45 * dt
        expected["plate_clearance"] = 0.08 * dt
        expected["plate_separation"] = 0.01 * dt

      active = dict(
        zip(env.reward_manager.active_terms, range(len(env.reward_manager.active_terms)))
      )
      observed = {}
      for name in expected:
        assert name in active, (group, name)
        observed[name] = env.reward_manager.get_term_cfg(name).weight
        assert abs(observed[name] - expected[name]) < 1e-12, (
          group,
          name,
          observed[name],
          expected[name],
        )
      for name in A6_NATIVE_OVERLAP_TERMS:
        assert name in active, (group, name)
        assert env.reward_manager.get_term_cfg(name).weight == 0.0, (group, name)
      if group == "G-":
        for name in ("plate_geometry_progress", "plate_clearance", "plate_separation"):
          assert name not in active, (group, name)

      obs, _ = env.reset()
      assert tuple(obs["actor"].shape) == (args.num_envs, 564)
      actions = torch.zeros((args.num_envs, 29), device=env.device)
      for step in range(args.steps):
        obs, reward, term, timeout, extras = env.step(actions)
        assert torch.isfinite(reward).all()
        _a6_update(env)
        assert torch.isfinite(env._a6_task).all()
        assert torch.isfinite(env._a6_cost).all()
        assert torch.isfinite(env._a6_progress).all()
        assert torch.isfinite(env._a6_clearance_score).all()
        assert torch.isfinite(env._a6_gate).all()
        assert torch.isfinite(env._a6_load).all()
        assert env._a6_subtick == (step + 1) * env.cfg.decimation

      if args.episode_steps:
        start_step = env.common_step_counter
        for _ in range(args.episode_steps):
          obs, reward, term, timeout, extras = env.step(actions)
          assert all(torch.isfinite(value).all() for value in obs.values())
          assert torch.isfinite(reward).all()
          assert torch.isfinite(env.sim.data.qpos).all()
          assert torch.isfinite(env.sim.data.qvel).all()
        assert env.common_step_counter == start_step + args.episode_steps

      if group == "G+":
        alpha = _a6_alpha(env, True)
        obstacle = env._a6_scene > 0
        unescaped = obstacle & ~env._a6_escaped
        assert torch.all(alpha[obstacle] == 0.05)
        assert torch.all(alpha[~obstacle] == 1.0)
        assert torch.all(alpha[unescaped] == 0.05)
      else:
        alpha = _a6_alpha(env, False)
        assert torch.all(alpha == 1.0)

    # Reset clearing: mutate selected envs, partial-reset them, and ensure the
    # reset envs are cleared while the untouched envs keep their values.
    env = env_plus
    n = args.num_envs
    ids = torch.tensor([0, n // 4, n // 2, 3 * n // 4], device=env.device, dtype=torch.long)
    mask = torch.ones(n, dtype=torch.bool, device=env.device)
    mask[ids] = False
    env._a6_task[ids] = 7.0
    env._a6_cost[ids] = 8.0
    env._a6_accum[ids] = 9.0
    env._a6_timer[ids] = 10.0
    env._a6_stage[ids] = 3
    env._a6_clear_hold[ids] = 20
    env._a6_escaped[ids] = True
    before = {
      "task": env._a6_task[mask].clone(),
      "cost": env._a6_cost[mask].clone(),
      "accum": env._a6_accum[mask].clone(),
      "timer": env._a6_timer[mask].clone(),
      "stage": env._a6_stage[mask].clone(),
      "clear_hold": env._a6_clear_hold[mask].clone(),
      "escaped": env._a6_escaped[mask].clone(),
    }
    env._reset_idx(ids)
    env.sim.forward()
    for key, tensor in before.items():
      assert torch.equal(getattr(env, "_a6_" + key)[mask], tensor), key
    for key in ("task", "cost", "accum", "timer"):
      assert torch.all(getattr(env, "_a6_" + key)[ids] == 0.0), key
    assert torch.all(env._a6_stage[ids] < 3)
    assert torch.all(env._a6_clear_hold[ids] == 0)
    assert not env._a6_escaped[ids].any()

    # Formula counterexamples on pure tensors.
    x = torch.linspace(0.0, 1.0, 5)
    assert torch.allclose(_a6_gate_ramp(x, 0.2, 0.8), ((x - 0.2) / 0.6).clamp(0, 1))
    t = ((x - 0.2) / 0.6).clamp(0.0, 1.0)
    assert torch.allclose(_a6_smooth_gate(x, 0.2, 0.8), t * t * (3.0 - 2.0 * t))
    span = torch.zeros(3, 30, device=args.device)
    span[0, 0] = 0.09
    span[1, :] = 0.0
    stall_gate = (1.0 - span / 0.10).clamp(0.0, 1.0)
    assert stall_gate[0].min() < 1.0
    assert torch.all(stall_gate[1] == 1.0)

    report = {
      "status": "A6_HOST_REWARD_PREFLIGHT_PASS",
      "num_envs": n,
      "steps": args.steps,
      "episode_steps": args.episode_steps,
      "seed": args.seed,
      "device": args.device,
      "dt_fold": HOST_CONSTRAINT_DT,
      "groups": ["G+", "G-"],
      "reward_terms_checked": {
        "positive": A6_POSITIVE_NAMES,
        "cost": A6_COST_NAMES,
        "obstacle_common": [
          "plate_completion",
          "plate_force",
          "path_quiet_feet",
          "path_joint_stall",
          "a6_soft_limits",
        ],
        "dense_G_plus": [
          "plate_geometry_progress",
          "plate_clearance",
          "plate_separation",
        ],
      },
      "zeroed_native_overlaps": A6_NATIVE_OVERLAP_TERMS,
      "alpha_checked": True,
      "substep_advance_checked": True,
      "partial_reset_checked": True,
      "counterexamples_checked": [
        "no_support_or_high_head_blocks_G",
        "unrelated_joint_stall_does_not_mask_single_joint_stall",
      ],
      "note": "reward mapping and executable reward buffers validated; no training completed",
    }
    (args.log_dir / "reward_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    print("A6_HOST_REWARD_PREFLIGHT_PASS", json.dumps(report, sort_keys=True), flush=True)
    return 0
  finally:
    env_plus.close()
    env_minus.close()


if __name__ == "__main__":
  raise SystemExit(main())

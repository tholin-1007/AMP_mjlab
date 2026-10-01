"""Preflight and train the paired historical-A6 HoST G-/G+ arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from src.tasks.host_recovery.a6 import a6_gminus_env_cfg, a6_gplus_env_cfg
from src.tasks.host_recovery.config.g1.rl_cfg import unitree_g1_host_standup_ppo_runner_cfg
from src.tasks.host_recovery.mdp.a6_runtime import scene_counts
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner


EXPECTED_BANKS = {
  "outputs/multiterrain_bank/train.npz": "287eae8e8840c1b3281e7010a84182b824fefacd34439c4f993e9f130af1027a",
  "datasets/reset_banks/natural_curriculum_v1/train.npz": "e9f94540520d7927dee01150a0f0bae39ad2aad5ae008b1c7c71f521650e44b9",
  "datasets/reset_banks/procedural_low_v1/train.npz": "e289bed8d1c93e87fbdcf969fc5fc741f2d8500a0908ed145d2a4e02e7fe116d",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def expected_counts(num_envs: int) -> dict[str, list[int]]:
  flat = num_envs // 2
  per_direction = flat // 4
  flat_low = round(per_direction * 0.4) * 4
  flat_late = round(per_direction * 0.2) * 4
  flat_middle = flat - flat_low - flat_late
  low = flat_low + num_envs // 2
  procedural = round(round(per_direction * 0.4) * 0.25) * 4 + round((num_envs // 16) * 0.25) * 8
  return {
    "scene": [flat, num_envs // 4, num_envs // 4, 0],
    "kind_low_middle_late": [low, flat_middle, flat_late],
    "natural_procedural": [num_envs - procedural, procedural],
    "direction": [num_envs // 4] * 4,
  }


def audit_reset(env: ManagerBasedRlEnv) -> dict:
  counts = scene_counts(env)
  expected = expected_counts(env.num_envs)
  if counts != expected:
    raise AssertionError((counts, expected))
  if float(env.sim.data.qvel.abs().max()) >= 1e-6:
    raise AssertionError("A6 reset velocities are not zero")
  if env.num_envs >= 32:
    ids = torch.tensor((0, env.num_envs // 4, env.num_envs // 2, 3 * env.num_envs // 4), device=env.device)
    untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    untouched[ids] = False
    before_qpos = env.sim.data.qpos[untouched].clone()
    before_mass = env.sim.model.body_mass[untouched].clone()
    before_gain = env._a6_gain_scale[untouched].clone()
    env._reset_idx(ids)
    assert torch.equal(before_qpos, env.sim.data.qpos[untouched])
    assert torch.equal(before_mass, env.sim.model.body_mass[untouched])
    assert torch.equal(before_gain, env._a6_gain_scale[untouched])
  return {"counts": counts, "partial_reset_untouched": env.num_envs - 4}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("gminus", "gplus"), required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--log-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--updates", type=int, default=10000)
  parser.add_argument("--seed", type=int, default=20261013)
  parser.add_argument("--forward-smoke", action="store_true")
  parser.add_argument("--preflight", action="store_true")
  args = parser.parse_args()

  args.log_dir.mkdir(parents=True, exist_ok=False)
  bank_hashes = {name: sha256(Path(name)) for name in EXPECTED_BANKS}
  assert bank_hashes == EXPECTED_BANKS, bank_hashes
  checkpoint_hash = sha256(args.checkpoint)
  cfg = (a6_gplus_env_cfg if args.arm == "gplus" else a6_gminus_env_cfg)()
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  agent = unitree_g1_host_standup_ppo_runner_cfg()
  agent.seed = args.seed
  agent.logger = "tensorboard"
  agent.experiment_name = f"g1_host_a6_{args.arm}"
  agent.max_iterations = args.updates
  agent.num_steps_per_env = 24
  agent.save_interval = 500
  dump_yaml(args.log_dir / "env.yaml", asdict(cfg))
  dump_yaml(args.log_dir / "agent.yaml", asdict(agent))

  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    assert abs(env.physics_dt - 0.002) < 1e-9
    assert abs(env.step_dt - 0.02) < 1e-9
    reset_audit = audit_reset(env)
    wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = HoSTOnPolicyRunner(wrapper, asdict(agent), str(args.log_dir), "cuda:0")
    runner.load(str(args.checkpoint), load_optimizer=True, map_location="cuda:0")
    # A6 is a new adaptation clock; retain native policy/normalizers/optimizer,
    # but do not inherit the flat pretraining update number or curricula time.
    runner.current_learning_iteration = 0
    env.common_step_counter = 0
    action = runner.get_inference_policy(device="cuda:0")
    observations = wrapper.get_observations()
    with torch.inference_mode():
      actions = action(observations["actor"])
    assert actions.shape == (args.num_envs, 29)
    assert torch.isfinite(actions).all()

    launch = {
      "arm": args.arm,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": checkpoint_hash,
      "method": "existing native 29-joint HoST port (single-critic PPO; not full paper HoST)",
      "checkpoint_state": "actor, critic, optimizer and observation normalizers retained",
      "adaptation_clock_reset": True,
      "num_envs": args.num_envs,
      "updates": args.updates,
      "seed": args.seed,
      "physics_dt": env.physics_dt,
      "control_dt": env.step_dt,
      "bank_sha256": bank_hashes,
      "reset_audit": reset_audit,
      "torch_compile_disabled": os.environ.get("TORCHDYNAMO_DISABLE") == "1",
      "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(),
    }
    (args.log_dir / "launch.json").write_text(json.dumps(launch, indent=2), encoding="utf-8")
    print("A6_FORWARD_SMOKE_OK", json.dumps(reset_audit), flush=True)
    if args.forward_smoke:
      return

    if args.preflight:
      # Cross one complete 10 s episode, then verify a reset has occurred and
      # all reward/observation channels remain finite.
      initial_counter = env.common_step_counter
      for _ in range(510):
        with torch.inference_mode():
          actions = action(observations["actor"])
          observations, reward, _, _ = wrapper.step(actions)
        assert torch.isfinite(reward).all()
        assert all(torch.isfinite(value).all() for value in observations.values())
      assert env.common_step_counter >= initial_counter + 510
      (args.log_dir / "preflight.json").write_text(
        json.dumps({"control_steps": 510, "crossed_full_episode": True}, indent=2),
        encoding="utf-8",
      )
      print("A6_FULL_EPISODE_PREFLIGHT_OK", flush=True)
      return

    runner.save(str(args.log_dir / "initial.pt"))
    runner.learn(num_learning_iterations=args.updates, init_at_random_ep_len=False)
    runner.save(str(args.log_dir / "final.pt"))
    print("A6_TRAINING_DONE", args.arm, flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()

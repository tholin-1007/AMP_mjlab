"""Runtime state for the frozen historical-A6 HoST adaptation.

This is deliberately self contained.  It translates the A6 bank/reset,
obstacle geometry, temporal state, shared task/cost terms and grouped dynamics
to the native HoST environment without importing SMP runtime code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields


TASK_NAMES = (
  "stage_pose", "head_velocity", "height", "upright", "feet_quiet",
  "base_quiet", "angular_quiet", "joint_quiet", "action_quiet",
)
TASK_WEIGHTS = (0.22, 0.18, 0.10, 0.15, 0.08, 0.07, 0.07, 0.06, 0.07)
COST_NAMES = (
  "action_rate", "action_acceleration", "joint_acceleration", "torque",
  "joint_overspeed", "joint_overpower", "head_overspeed",
  "sustained_effort", "soft_joint_limits", "quiet_feet", "joint_stall",
)
COST_WEIGHTS = (-0.0015, -0.0012, -5e-8, -1e-6, -0.02, -2e-6, -1.0, -0.05, -0.10, -0.03, -0.20)
ESCAPE_NAMES = ("geometry_progress", "dense_clearance", "completion", "separation", "excess_force")
ESCAPE_WEIGHTS = (0.45, 0.08, 0.60, 0.01, -0.03)
DIRECTIONS = ("supine", "prone", "left_side_down", "right_side_down")


def gate(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
  return ((value - low) / (high - low)).clamp(0.0, 1.0)


def _natural_weights(bank: Any, stage: str) -> np.ndarray:
  """Historical clip-balanced weights restricted to one reset stage."""
  rows = np.flatnonzero(bank["stages"] == stage)
  groups = np.asarray([x.replace("__mirror", "") for x in bank["clips"]])
  weights = np.zeros(len(bank["qpos"]), dtype=np.float64)
  clips = sorted(set(groups[rows]))
  for clip in clips:
    subset = rows[groups[rows] == clip]
    weights[subset] = 1.0 / len(clips) / len(subset)
  return weights


def _effort_limits(robot: Any) -> torch.Tensor:
  chunks = [act.default_force_limit[0].reshape(-1) for act in robot.actuators]
  result = torch.cat(chunks)
  if result.numel() != robot.data.joint_pos.shape[-1]:
    raise RuntimeError("A6 actuator/joint effort-limit ordering mismatch")
  return result


def initialize(env: Any) -> None:
  if hasattr(env, "_a6_scene"):
    return
  n, device = env.num_envs, env.device
  if n < 32 or n % 32:
    raise ValueError("historical A6 requires num_envs >= 32 and divisible by 32")

  bank = np.load("outputs/multiterrain_bank/train.npz", allow_pickle=False)
  natural = np.load("datasets/reset_banks/natural_curriculum_v1/train.npz", allow_pickle=False)
  env._a6_low_bank = torch.as_tensor(bank["qpos"], device=device, dtype=torch.float32)
  env._a6_natural_bank = torch.as_tensor(natural["qpos"], device=device, dtype=torch.float32)

  counts = (n // 2, n // 4, n // 4)
  scene = np.repeat(np.arange(3), counts)
  direction = np.concatenate([np.repeat(np.arange(4), count // 4) for count in counts])
  env._a6_scene = torch.as_tensor(scene, device=device)
  env._a6_direction = torch.as_tensor(direction, device=device)
  env._a6_kind = torch.zeros(n, device=device, dtype=torch.long)  # low/middle/late
  env._a6_source = torch.zeros(n, device=device, dtype=torch.long)  # natural/procedural
  for direction_id in range(4):
    flat = torch.where((env._a6_scene == 0) & (env._a6_direction == direction_id))[0]
    low = round(len(flat) * 0.4)
    late = round(len(flat) * 0.2)
    env._a6_kind[flat[low:len(flat) - late]] = 1
    env._a6_kind[flat[len(flat) - late:]] = 2
    env._a6_source[flat[:round(low * 0.25)]] = 1
  obstacle_low = torch.where(env._a6_scene > 0)[0]
  for scene_id in (1, 2):
    for direction_id in range(4):
      subset = obstacle_low[(env._a6_scene[obstacle_low] == scene_id) & (env._a6_direction[obstacle_low] == direction_id)]
      env._a6_source[subset[:round(len(subset) * 0.25)]] = 1

  env._a6_low_pools = {}
  for scene_id in range(3):
    for direction_id in range(4):
      for source_id in range(2):
        rows = np.flatnonzero(
          (bank["stratum"] == scene_id)
          & (bank["direction"] == direction_id)
          & (bank["source"] == source_id)
        )
        if not len(rows):
          raise RuntimeError(f"empty A6 low-bank pool {(scene_id, direction_id, source_id)}")
        env._a6_low_pools[(scene_id, direction_id, source_id)] = torch.as_tensor(rows, device=device)
  env._a6_natural_pools = []
  for kind, stage in ((1, "middle"), (2, "late")):
    weights = _natural_weights(natural, stage)
    rows = np.flatnonzero(weights > 0)
    env._a6_natural_pools.append((
      kind,
      torch.as_tensor(rows, device=device),
      torch.as_tensor(weights[rows] / weights[rows].sum(), device=device, dtype=torch.float32),
    ))

  env._a6_rng = torch.Generator(device=device).manual_seed(int(env.cfg.seed) + 91019)
  env._a6_dynamics_rng = torch.Generator(device=device).manual_seed(int(env.cfg.seed) + 131071)
  robot = env.scene["robot"]
  env._a6_effort_limits = _effort_limits(robot).to(device)
  env._a6_command_lag = torch.zeros(n, dtype=torch.long, device=device)
  env._a6_mass_scale = torch.ones(n, 4, device=device)
  env._a6_gain_scale = torch.ones(n, 6, device=device)
  env._a6_nominal = torch.ones(n, dtype=torch.bool, device=device)

  model = env.sim.mj_model
  body_names = [model.body(int(body)).name.split("/")[-1] for body in robot.indexing.body_ids]
  def descendants(name: str) -> list[int]:
    root = model.body("robot/" + name).id
    output = []
    for local_id, body_id in enumerate(robot.indexing.body_ids.tolist()):
      cursor = body_id
      while cursor > 0 and cursor != root:
        cursor = int(model.body_parentid[cursor])
      if cursor == root:
        output.append(local_id)
    return output
  legs = set(descendants("left_hip_pitch_link") + descendants("right_hip_pitch_link"))
  upper = set(descendants("torso_link"))
  env._a6_body_groups = torch.tensor(
    [0 if i in legs else 1 if i in upper else 2 for i in range(len(body_names))],
    device=device,
  )
  env._a6_body_ids = robot.indexing.body_ids.long()
  env._a6_actuator_groups = []
  for actuator in robot.actuators:
    name = actuator._target_names[0]
    group = 0 if "waist" in name else 1 if "hip" in name else 2 if "knee" in name else 3 if "ankle" in name else 4 if ("shoulder" in name or "elbow" in name) else 5
    env._a6_actuator_groups.append(group)

  collision_ids, _ = robot.find_geoms(".*_collision")
  env._a6_robot_geoms = robot.indexing.geom_ids[torch.as_tensor(collision_ids, device=device)].long()
  env._a6_plate_geom_ids = []
  for entity_name, geom_name in (("escape_obstacle", "escape_plate_geom"), ("free_obstacle", "plate_geom")):
    entity = env.scene[entity_name]
    local_ids, _ = entity.find_geoms((geom_name,))
    env._a6_plate_geom_ids.append(entity.indexing.geom_ids[local_ids[0]].long())

  joint_shape = robot.data.joint_pos.shape
  env._a6_stage = torch.zeros(n, dtype=torch.long, device=device)
  env._a6_stage_hold = torch.zeros(n, dtype=torch.long, device=device)
  env._a6_task = torch.zeros(n, len(TASK_NAMES), device=device)
  env._a6_cost = torch.zeros(n, len(COST_NAMES), device=device)
  env._a6_substep_cost = torch.zeros(n, 7, device=device)
  env._a6_load_timer = torch.zeros(joint_shape, device=device)
  env._a6_joint_load_acc = torch.zeros(joint_shape, device=device)
  env._a6_joint_motion = torch.zeros(25, *joint_shape, device=device)
  env._a6_motion_cursor = 0
  env._a6_last_action = torch.zeros(joint_shape, device=device)
  env._a6_last_delta = torch.zeros(joint_shape, device=device)
  env._a6_prev_target = torch.zeros(joint_shape, device=device)
  env._a6_ever_contact = torch.zeros(n, dtype=torch.bool, device=device)
  env._a6_clear_hold = torch.zeros(n, dtype=torch.long, device=device)
  env._a6_escaped = torch.zeros(n, dtype=torch.bool, device=device)
  env._a6_invalid = torch.zeros(n, dtype=torch.bool, device=device)
  for name in ("best_score", "best_clearance", "initial_count", "geometry_progress", "dense_clearance", "separation", "best_distance", "force", "support"):
    setattr(env, "_a6_" + name, torch.zeros(n, device=device))
  env._a6_control_tick = -1
  env._a6_subtick = 0


def robot_bounds(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
  ids = env._a6_robot_geoms
  data, model = env.sim.data, env.sim.model
  position = data.geom_xpos[:, ids]
  rotation = data.geom_xmat[:, ids]
  size = model.geom_size[:, ids]
  geom_type = model.geom_type[ids]
  extent = torch.einsum("ngij,ngj->ngi", rotation.abs(), size)
  sphere = geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
  capsule = geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  extent = torch.where(sphere[None, :, None], size[:, :, 0, None], extent)
  extent = torch.where(capsule[None, :, None], size[:, :, 0, None] + size[:, :, 1, None] * rotation[:, :, :, 2].abs(), extent)
  return position, extent


def geometry(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  position, extent = robot_bounds(env)
  rows = torch.arange(env.num_envs, device=env.device)
  plate_ids = torch.stack(env._a6_plate_geom_ids)[(env._a6_scene == 2).long()]
  plate_position = env.sim.data.geom_xpos[rows, plate_ids]
  plate_rotation = env.sim.data.geom_xmat[rows, plate_ids]
  plate_size = env.sim.model.geom_size[rows, plate_ids]
  plate_extent = torch.einsum("nij,nj->ni", plate_rotation.abs(), plate_size)
  overlap = extent[..., :2] + plate_extent[:, None, :2] - (position[..., :2] - plate_position[:, None, :2]).abs()
  covered = (overlap > 0).all(-1)
  score = torch.where(covered, overlap.clamp_min(0).amin(-1), 0.0).sum(-1)
  clearance = torch.where(covered, 0.0, (-overlap).clamp_min(0).norm(dim=-1)).amin(-1)
  return covered.sum(-1), score, clearance


def _write_bank_state(env: Any, ids: torch.Tensor, qpos: torch.Tensor) -> None:
  robot = env.scene["robot"]
  state = robot.data.default_root_state[ids].clone()
  state[:, :3] = qpos[:, :3] + env.scene.env_origins[ids]
  state[:, 3:7] = qpos[:, 3:7]
  state[:, 7:] = 0.0
  robot.write_root_state_to_sim(state, env_ids=ids)
  robot.write_joint_state_to_sim(qpos[:, 7:], torch.zeros_like(qpos[:, 7:]), env_ids=ids)


def _sample_reset_bank(env: Any, ids: torch.Tensor) -> None:
  low_ids = ids[env._a6_kind[ids] == 0]
  if len(low_ids):
    sampled = torch.empty(len(low_ids), dtype=torch.long, device=env.device)
    for key, pool in env._a6_low_pools.items():
      mask = (
        (env._a6_scene[low_ids] == key[0])
        & (env._a6_direction[low_ids] == key[1])
        & (env._a6_source[low_ids] == key[2])
      )
      count = int(mask.sum())
      if count:
        sampled[mask] = pool[torch.randint(len(pool), (count,), generator=env._a6_rng, device=env.device)]
    _write_bank_state(env, low_ids, env._a6_low_bank[sampled])
  for kind, pool, weights in env._a6_natural_pools:
    target = ids[env._a6_kind[ids] == kind]
    if len(target):
      draw = pool[torch.multinomial(weights, len(target), replacement=True, generator=env._a6_rng)]
      _write_bank_state(env, target, env._a6_natural_bank[draw])


def _randomize_dynamics(env: Any, ids: torch.Tensor, enabled: bool) -> None:
  count = len(ids)
  width = 0.1 + 0.1 * min(env.common_step_counter / (24 * 2000), 1.0)
  rand = lambda shape: torch.rand(shape, generator=env._a6_dynamics_rng, device=env.device)
  active = rand((count,)) >= 0.25 if enabled else torch.zeros(count, dtype=torch.bool, device=env.device)
  mass = 1.0 + (2.0 * rand((count, 4)) - 1.0) * width
  gain = 1.0 + (2.0 * rand((count, 6)) - 1.0) * width
  mass[~active] = 1.0
  gain[~active] = 1.0
  mass[:, 3] = 1.0
  env._a6_mass_scale[ids] = mass
  env._a6_gain_scale[ids] = gain
  env._a6_nominal[ids] = ~active
  lag = torch.randint(0, 6, (count,), generator=env._a6_dynamics_rng, device=env.device)
  env._a6_command_lag[ids] = torch.where(active, lag, 0)
  bodies = env._a6_body_ids
  scale = mass[:, env._a6_body_groups]
  default_mass = env.sim.get_default_field("body_mass")[bodies]
  default_inertia = env.sim.get_default_field("body_inertia")[bodies]
  env.sim.model.body_mass[ids[:, None], bodies[None, :]] = default_mass[None] * scale
  env.sim.model.body_inertia[ids[:, None], bodies[None, :]] = default_inertia[None] * scale[:, :, None]
  for actuator, group in zip(env.scene["robot"].actuators, env._a6_actuator_groups, strict=True):
    actuator.set_gains(
      ids,
      kp=actuator.default_stiffness[ids] * gain[:, group, None],
      kd=actuator.default_damping[ids] * gain[:, group, None],
    )


@requires_model_fields("body_mass", "body_inertia", "geom_size", recompute=RecomputeLevel.set_const)
def reset(env: Any, env_ids: torch.Tensor | None = None, dynamics: bool = True, evaluation: bool = False) -> None:
  initialize(env)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  _randomize_dynamics(env, ids, enabled=dynamics and not evaluation)
  _sample_reset_bank(env, ids)
  env.sim.forward()

  robot = env.scene["robot"]
  position, extent = robot_bounds(env)
  parked = robot.data.default_root_state[ids, :7].clone()
  parked[:, :3] = env.scene.env_origins[ids] + parked.new_tensor((20.0, 20.0, 0.1))
  parked[:, 3:] = parked.new_tensor((1.0, 0.0, 0.0, 0.0))
  guided, free = env.scene["escape_obstacle"], env.scene["free_obstacle"]
  guided.write_mocap_pose_to_sim(parked, env_ids=ids)
  guided.write_joint_state_to_sim(torch.zeros(len(ids), 1, device=env.device), torch.zeros(len(ids), 1, device=env.device), env_ids=ids)
  free_state = free.data.default_root_state[ids].clone()
  free_state[:, :7] = parked
  free_state[:, 0] += 2.0
  free_state[:, 7:] = 0.0
  free.write_root_state_to_sim(free_state, env_ids=ids)
  env.sim.forward()

  board = parked.clone()
  board[:, :2] = robot.data.root_link_pos_w[ids, :2]
  covered = ((position[ids, :, :2] - board[:, None, :2]).abs() < extent[ids, :, :2] + board.new_tensor((0.45, 0.32))).all(-1)
  top = torch.where(covered, position[ids, :, 2] + extent[ids, :, 2], -torch.inf).amax(-1)
  board[:, 2] = top + 0.035 + 0.002
  for scene_id, entity in ((1, guided), (2, free)):
    choose = env._a6_scene[ids] == scene_id
    entity_ids = ids[choose]
    if not len(entity_ids):
      continue
    if scene_id == 1:
      entity.write_mocap_pose_to_sim(board[choose], env_ids=entity_ids)
    else:
      state = entity.data.default_root_state[entity_ids].clone()
      state[:, :7] = board[choose]
      state[:, 7:] = 0.0
      entity.write_root_state_to_sim(state, env_ids=entity_ids)
    body_id = entity.indexing.body_ids[-1].long()
    mass = torch.full((len(entity_ids),), 6.0, device=env.device) if evaluation else 4.0 + torch.rand(len(entity_ids), generator=env._a6_rng, device=env.device) * (2.0 + 6.0 * min(env.common_step_counter / 100000, 1.0))
    default_mass = env.sim.get_default_field("body_mass")[body_id]
    default_inertia = env.sim.get_default_field("body_inertia")[body_id]
    env.sim.model.body_mass[entity_ids, body_id] = mass
    env.sim.model.body_inertia[entity_ids, body_id] = default_inertia * mass[:, None] / default_mass
  env.sim.forward()

  count, score, clearance = geometry(env)
  env._a6_best_score[ids] = score[ids]
  env._a6_best_clearance[ids] = clearance[ids]
  env._a6_initial_count[ids] = count[ids].float()
  env._a6_best_distance[ids] = 0.0
  env._a6_ever_contact[ids] = False
  env._a6_clear_hold[ids] = 0
  env._a6_escaped[ids] = False
  env._a6_invalid[ids] = False
  for name in ("geometry_progress", "dense_clearance", "separation", "force", "support"):
    getattr(env, "_a6_" + name)[ids] = 0.0
  env._a6_load_timer[ids] = 0.0
  env._a6_joint_load_acc[ids] = 0.0
  env._a6_joint_motion[:, ids] = robot.data.joint_pos[ids][None]
  env._a6_last_action[ids] = 0.0
  env._a6_last_delta[ids] = 0.0
  env._a6_prev_target[ids] = robot.data.joint_pos[ids]
  env._a6_stage_hold[ids] = 0
  z = torso_height(env)[ids]
  upright = (-robot.data.projected_gravity_b[ids, 2]).clamp(0.0, 1.0)
  stage = torch.zeros_like(ids)
  stage = torch.where((z >= 0.55) & (upright >= 0.55), 1, stage)
  stage = torch.where((z >= 0.78) & (upright >= 0.72), 2, stage)
  stage = torch.where((z >= 1.08) & (upright >= 0.85), 3, stage)
  env._a6_stage[ids] = stage
  env._a6_control_tick = -1


def torso_height(env: Any) -> torch.Tensor:
  robot = env.scene["robot"]
  torso_id = robot.find_bodies("torso_link")[0][0]
  return robot.data.body_link_pos_w[:, torso_id, 2] - env.scene.env_origins[:, 2] + 0.43


def _contact_state(env: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  free_scene = env._a6_scene == 2
  contacts, forces, depths = [], [], []
  for name in ("guided_contact", "free_contact"):
    data = env.scene[name].data
    contacts.append((data.found > 0).any(-1))
    forces.append(data.force.norm(dim=-1).amax(-1))
    depths.append(data.dist.amin(-1))
  return (
    torch.where(free_scene, contacts[1], contacts[0]),
    torch.where(free_scene, forces[1], forces[0]),
    torch.where(free_scene, depths[1], depths[0]),
  )


def sample_substep(env: Any) -> torch.Tensor:
  initialize(env)
  if env._a6_subtick % env.cfg.decimation == 0:
    env._a6_substep_cost.zero_()
    env._a6_joint_load_acc.zero_()
  robot = env.scene["robot"]
  torque, velocity = robot.data.qfrc_actuator, robot.data.joint_vel
  stage = env._a6_stage
  ratio = torque / env._a6_effort_limits
  high = ratio.abs() > 0.7
  env._a6_load_timer = torch.where(high, env._a6_load_timer + env.physics_dt, torch.zeros_like(env._a6_load_timer))
  stall = gate(ratio.abs(), 0.7, 0.95).square() * gate(env._a6_load_timer, 0.3, 1.0) * gate(0.3 - velocity.abs(), 0.0, 0.3)
  env._a6_joint_load_acc += stall
  speed_limit = velocity.new_tensor((6.0, 5.0, 4.0, 3.5))[stage, None]
  power_limit = velocity.new_tensor((140.0, 110.0, 90.0, 75.0))[stage, None]
  effort = ((ratio.square().sqrt() - 0.45).clamp_min(0.0) / 0.55).square().topk(3, dim=-1).values.mean(-1)
  vz = robot.data.root_link_lin_vel_w[:, 2]
  env._a6_substep_cost += torch.stack((
    robot.data.joint_acc.square().sum(-1) * velocity.new_tensor((0.35, 0.65, 1.0, 1.0))[stage],
    torque.square().sum(-1) * velocity.new_tensor((0.5, 0.75, 1.0, 1.0))[stage],
    (velocity.abs() - speed_limit).clamp_min(0.0).square().sum(-1),
    ((torque * velocity).abs() - power_limit).clamp_min(0.0).square().mean(-1),
    (vz - velocity.new_tensor((0.30, 0.30, 0.30, 0.20))[stage]).clamp_min(0.0).square() + (-vz - 0.20).clamp_min(0.0).square(),
    effort,
    stall.amax(-1),
  ), dim=-1)
  env._a6_subtick += 1
  return stall.amax(-1)


def update(env: Any) -> None:
  initialize(env)
  if env._a6_control_tick == env.common_step_counter:
    return
  env._a6_control_tick = env.common_step_counter
  active = env._a6_scene > 0
  contact, force, depth = _contact_state(env)
  env._a6_force.copy_(force)
  env._a6_ever_contact |= active & contact
  count, score, clearance = geometry(env)
  score_delta = (env._a6_best_score - score).clamp_min(0.0)
  clearance_delta = (clearance - env._a6_best_clearance).clamp_min(0.0)
  env._a6_best_score = torch.minimum(env._a6_best_score, score)
  env._a6_best_clearance = torch.maximum(env._a6_best_clearance, clearance)
  support = (env.scene["path_hands"].data.found > 0).float().mean(-1)
  env._a6_support.copy_(support)
  z = torso_height(env)
  eligible = active & env._a6_ever_contact & ~env._a6_escaped & ~env._a6_invalid
  env._a6_geometry_progress = eligible * (z <= 0.90) * support * ((score_delta / 0.025).clamp(0.0, 1.0) + 0.5 * (clearance_delta / 0.02).clamp(0.0, 1.0))
  env._a6_dense_clearance = active * ~env._a6_invalid * (0.85 * (1.0 - count / env._a6_initial_count.clamp_min(1.0)).clamp(0.0, 1.0) + 0.15 * (clearance / 0.04).clamp(0.0, 1.0))
  rows = torch.arange(env.num_envs, device=env.device)
  plate_ids = torch.stack(env._a6_plate_geom_ids)[(env._a6_scene == 2).long()]
  distance = (env.scene["robot"].data.root_link_pos_w[:, :2] - env.sim.data.geom_xpos[rows, plate_ids, :2]).norm(dim=-1)
  env._a6_separation = torch.where(active & env._a6_ever_contact, ((distance - env._a6_best_distance) / 0.025).clamp(0.0, 1.0), 0.0)
  env._a6_best_distance = torch.maximum(env._a6_best_distance, distance)
  clear = active & env._a6_ever_contact & ~contact & (clearance >= 0.025)
  env._a6_clear_hold = torch.where(clear, env._a6_clear_hold + 1, 0)
  env._a6_escaped = env._a6_clear_hold >= 15
  env._a6_invalid = active & ((depth < -0.02) | (force > 1500.0) | ((env.episode_length_buf > 25) & ~env._a6_ever_contact))
  _update_shared_terms(env, z)


def _update_shared_terms(env: Any, z: torch.Tensor) -> None:
  robot = env.scene["robot"]
  stage = env._a6_stage
  upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
  velocity_z = robot.data.root_link_lin_vel_w[:, 2]
  knee_ids, _ = robot.find_joints(("left_knee_joint", "right_knee_joint"))
  knees = robot.data.joint_pos[:, knee_ids]
  fallen = (z < 0.65) & (upright < 0.45)
  stage[fallen] = 0
  env._a6_stage_hold[fallen] = 0
  feet_force = env.scene["quality_feet"].data.force[..., 2].abs().amin(-1)
  ready = (
    ((stage == 0) & (z >= 0.55) & (upright >= 0.55) & (knees.amin(-1) >= 0.8) & (velocity_z.abs() <= 0.16))
    | ((stage == 1) & (z >= 0.78) & (upright >= 0.72) & (knees.amin(-1) >= 0.6) & (velocity_z.abs() <= 0.18))
    | ((stage == 2) & (z >= 1.08) & (upright >= 0.85) & (knees.abs().amax(-1) < 0.8) & (feet_force > 20.0) & (velocity_z.abs() <= 0.12))
  )
  env._a6_stage_hold = torch.where(ready, env._a6_stage_hold + 1, 0)
  advance = env._a6_stage_hold >= torch.where(stage == 2, 25, 10)
  stage.copy_(torch.where(advance, (stage + 1).clamp_max(3), stage))
  env._a6_stage_hold[advance] = 0
  target_height = z.new_tensor((0.62, 0.86, 1.15, 1.15))[stage]
  target_upright = z.new_tensor((0.60, 0.76, 0.93, 0.93))[stage]
  knee_low = z.new_tensor((0.8, 0.6, 0.0, 0.0))[stage, None]
  knee_high = z.new_tensor((1.8, 1.6, 0.65, 0.65))[stage, None]
  knee_error = ((knee_low - knees).clamp_min(0.0).square() + (knees - knee_high).clamp_min(0.0).square()).mean(-1)
  pose = torch.exp(-8.0 * (target_height - z).clamp_min(0.0).square() - 6.0 * (target_upright - upright).clamp_min(0.0).square() - 5.0 * knee_error)
  target_velocity = z.new_tensor((0.10, 0.15, 0.20, 0.0))[stage] * ((target_height - z) / 0.2).clamp(0.0, 1.0)
  up_limit = z.new_tensor((0.25, 0.30, 0.30, 0.12))[stage]
  down_limit = z.new_tensor((0.16, 0.18, 0.18, 0.12))[stage]
  excess = (velocity_z - up_limit).clamp_min(0.0).square() + (-velocity_z - down_limit).clamp_min(0.0).square()
  velocity = torch.exp(-45.0 * (velocity_z - target_velocity).square() - 140.0 * excess)
  quiet = gate(z, 0.85, 1.15) * gate(upright, 0.70, 0.93)
  foot_ids = [robot.find_bodies(name)[0][0] for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
  foot_motion = robot.data.body_link_lin_vel_w[:, foot_ids].square().sum(-1).mean(-1)
  action = env.action_manager.action
  delta = action - env._a6_last_action
  acceleration = delta - env._a6_last_delta
  fresh = env.episode_length_buf <= 1
  delta = torch.where(fresh[:, None], 0.0, delta)
  acceleration = torch.where(fresh[:, None], 0.0, acceleration)
  env._a6_task.copy_(torch.stack((
    pose,
    velocity,
    torch.exp(-2.0 * (1.15 - z).clamp_min(0.0).square()),
    upright.square(),
    1.0 - quiet + quiet * torch.exp(-20.0 * foot_motion),
    1.0 - quiet + quiet * torch.exp(-8.0 * robot.data.root_link_lin_vel_w.square().sum(-1)),
    torch.exp(-0.8 * robot.data.root_link_ang_vel_w.square().sum(-1)),
    torch.exp(-0.04 * robot.data.joint_vel.square().mean(-1)),
    torch.exp(-4.0 * delta.square().mean(-1)),
  ), dim=-1))
  env._a6_cost[:, 0] = delta.square().sum(-1) * z.new_tensor((0.5, 0.75, 1.0, 1.0))[stage]
  env._a6_cost[:, 1] = acceleration.square().sum(-1) * z.new_tensor((0.3, 0.6, 1.0, 1.0))[stage]
  env._a6_cost[:, 2:8] = env._a6_substep_cost[:, :6] / env.cfg.decimation
  limits = robot.data.soft_joint_pos_limits
  env._a6_cost[:, 8] = ((limits[..., 0] - robot.data.joint_pos).clamp_min(0.0) + (robot.data.joint_pos - limits[..., 1]).clamp_min(0.0)).sum(-1)
  env._a6_cost[:, 9] = quiet * (foot_motion / 0.1**2).clamp(0.0, 10.0)
  env._a6_joint_motion[env._a6_motion_cursor] = robot.data.joint_pos
  env._a6_motion_cursor = (env._a6_motion_cursor + 1) % 25
  span = env._a6_joint_motion.amax(0) - env._a6_joint_motion.amin(0)
  stall_gate = (1.0 - span / 0.10).clamp(0.0, 1.0)
  env._a6_cost[:, 10] = (env._a6_joint_load_acc / env.cfg.decimation * stall_gate).amax(-1)
  env._a6_last_action.copy_(action)
  env._a6_last_delta.copy_(delta)


def task_reward(env: Any, index: int, dense_guidance: bool) -> torch.Tensor:
  update(env)
  constrained = (env._a6_scene > 0) & ~env._a6_escaped
  alpha = torch.where(constrained, 0.05, 1.0) if dense_guidance else torch.ones(env.num_envs, device=env.device)
  return alpha * env._a6_task[:, index]


def cost_reward(env: Any, index: int) -> torch.Tensor:
  update(env)
  return env._a6_cost[:, index]


def escape_reward(env: Any, index: int, dense_guidance: bool) -> torch.Tensor:
  update(env)
  if index == 0:
    return env._a6_geometry_progress if dense_guidance else torch.zeros_like(env._a6_geometry_progress)
  if index == 1:
    return env._a6_dense_clearance if dense_guidance else torch.zeros_like(env._a6_dense_clearance)
  if index == 2:
    return env._a6_escaped.float()
  if index == 3:
    return env._a6_separation if dense_guidance else torch.zeros_like(env._a6_separation)
  return (env._a6_scene > 0).float() * ((env._a6_force - 300.0).clamp_min(0.0) / 300.0).square()


def invalid_plate(env: Any) -> torch.Tensor:
  update(env)
  return env._a6_invalid


def scene_counts(env: Any) -> dict[str, list[int]]:
  initialize(env)
  return {
    "scene": torch.bincount(env._a6_scene, minlength=4).tolist(),
    "kind_low_middle_late": torch.bincount(env._a6_kind, minlength=3).tolist(),
    "natural_procedural": torch.bincount(env._a6_source, minlength=2).tolist(),
    "direction": torch.bincount(env._a6_direction, minlength=4).tolist(),
  }

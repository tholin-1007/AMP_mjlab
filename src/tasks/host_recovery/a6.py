"""Historical A6 G-/G+ paired adaptation for the native 29-joint HoST port."""

from __future__ import annotations

import copy
from dataclasses import fields

from mjlab.entity import EntityCfg
from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.host_recovery import mdp
from src.tasks.host_recovery.flat29 import flat29_env_cfg


CONTROL_DT = 0.02


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
  ground = ContactMatch(mode="geom", pattern="terrain")
  return (
    ContactSensorCfg(
      name="quality_feet",
      primary=ContactMatch(
        mode="body",
        pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
        entity="robot",
      ),
      secondary=ground,
      fields=("force",),
      reduce="netforce",
    ),
    ContactSensorCfg(
      name="quality_other",
      primary=ContactMatch(
        mode="geom", pattern=r"^(?!(left|right)_foot[1-7]_collision$).*_collision$", entity="robot"
      ),
      secondary=ground,
      fields=("force",),
      reduce="netforce",
    ),
    ContactSensorCfg(
      name="path_hands",
      primary=ContactMatch(
        mode="geom", pattern=r"(left|right)_hand_collision$", entity="robot"
      ),
      secondary=ground,
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="guided_contact",
      primary=ContactMatch(mode="geom", pattern=r".*_collision$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="escape_plate", entity="escape_obstacle"),
      fields=("found", "force", "dist"),
      reduce="mindist",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="free_contact",
      primary=ContactMatch(mode="geom", pattern=r".*_collision$", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="plate", entity="free_obstacle"),
      fields=("found", "force", "dist"),
      reduce="mindist",
      num_slots=1,
    ),
  )


def a6_env_cfg(*, dense_guidance: bool, play: bool = False):
  cfg = flat29_env_cfg(play=False)
  cfg.scene.num_envs = 32 if play else 4096
  cfg.scene.extent = 2.0
  cfg.scene.env_spacing = 0.0
  cfg.scene.entities["escape_obstacle"] = EntityCfg(
    spec_fn=mdp.guided_plate_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(20.0, 20.0, 0.8),
      joint_pos={"escape_plate_slide": 0.0},
      joint_vel={"escape_plate_slide": 0.0},
    ),
  )
  cfg.scene.entities["free_obstacle"] = EntityCfg(
    spec_fn=mdp.free_plate_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(20.0, 20.0, 0.1)),
  )
  cfg.scene.sensors = tuple(cfg.scene.sensors) + _contact_sensors()

  # The bank event owns all robot/plate/dynamics reset state.  Keeping either
  # generic HoST pose reset would overwrite the frozen qpos after it ran.
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_joints", None)
  cfg.events.pop("init_pull_force", None)
  cfg.events.pop("apply_pull_force", None)
  cfg.events["a6_reset"] = EventTermCfg(
    func=mdp.reset,
    mode="reset",
    params={"dynamics": not play, "evaluation": play},
  )
  cfg.events["push_robot"] = EventTermCfg(
    func=push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
      }
    },
  )
  cfg.curriculum = {}

  old_action = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = mdp.A6DelayedIncrementalJointPositionActionCfg(
    **{field.name: getattr(old_action, field.name) for field in fields(old_action)}
  )
  cfg.actions["joint_pos"].scale = 0.25

  # Common costs replace native terms with the same mathematical role.  HoST
  # task/style/target terms stay as its disclosed native prior/mechanism.
  for duplicate in (
    "regu_dof_acc", "regu_action_rate", "regu_smoothness", "regu_dof_vel",
    "regu_upper_dof_vel", "regu_dof_pos_limits",
  ):
    cfg.rewards.pop(duplicate, None)
  for index, (name, weight) in enumerate(zip(mdp.TASK_NAMES, mdp.TASK_WEIGHTS, strict=True)):
    cfg.rewards["a6_task_" + name] = RewardTermCfg(
      func=mdp.task_reward,
      params={"index": index, "dense_guidance": dense_guidance},
      weight=weight * CONTROL_DT,
    )
  for index, (name, weight) in enumerate(zip(mdp.COST_NAMES, mdp.COST_WEIGHTS, strict=True)):
    cfg.rewards["a6_cost_" + name] = RewardTermCfg(
      func=mdp.cost_reward,
      params={"index": index},
      weight=weight * CONTROL_DT,
    )
  for index, (name, weight) in enumerate(zip(mdp.ESCAPE_NAMES, mdp.ESCAPE_WEIGHTS, strict=True)):
    cfg.rewards["a6_escape_" + name] = RewardTermCfg(
      func=mdp.escape_reward,
      params={"index": index, "dense_guidance": dense_guidance},
      weight=weight * CONTROL_DT,
    )
  cfg.terminations["invalid_plate"] = TerminationTermCfg(func=mdp.invalid_plate)
  cfg.metrics["a6_substep"] = MetricsTermCfg(func=mdp.sample_substep, per_substep=True)

  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.iterations = 100
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 4000
  cfg.decimation = 10
  cfg.episode_length_s = 10.0
  cfg.scale_rewards_by_dt = False
  cfg.seed = 20261013
  if play:
    cfg.observations["actor"].terms["host"].params["add_noise"] = False
    cfg.events.pop("push_robot", None)
  return cfg


def a6_gminus_env_cfg(play: bool = False):
  return a6_env_cfg(dense_guidance=False, play=play)


def a6_gplus_env_cfg(play: bool = False):
  return a6_env_cfg(dense_guidance=True, play=play)

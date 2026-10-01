"""Historical A6 per-environment command delay for native HoST actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .actions import IncrementalJointPositionAction, IncrementalJointPositionActionCfg
from .observations import UNACTUATED_STEPS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class A6DelayedIncrementalJointPositionActionCfg(IncrementalJointPositionActionCfg):
  """Incremental HoST target with a 0--10 ms physics-substep delay."""

  def build(self, env: "ManagerBasedRlEnv") -> "A6DelayedIncrementalJointPositionAction":
    return A6DelayedIncrementalJointPositionAction(self, env)


class A6DelayedIncrementalJointPositionAction(IncrementalJointPositionAction):
  """Apply the processed target selected from a six-entry ring buffer.

  ``apply_actions`` is called once per 2 ms physics step.  A lag value in
  ``[0, 5]`` therefore represents exactly ``0, 2, 4, 6, 8, 10`` ms.
  """

  def __init__(self, cfg: A6DelayedIncrementalJointPositionActionCfg, env: "ManagerBasedRlEnv") -> None:
    super().__init__(cfg, env)
    self._a6_env = env
    self._a6_buffer = self._processed_actions.unsqueeze(0).repeat(6, 1, 1)
    self._a6_cursor = 0
    self._a6_rows = torch.arange(self.num_envs, device=self.device)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    if hasattr(self, "_a6_buffer"):
      current = self._entity.data.joint_pos[ids][:, self._target_ids]
      self._a6_buffer[:, ids] = current.unsqueeze(0)

  def apply_actions(self) -> None:
    lag = getattr(
      self._a6_env,
      "_a6_command_lag",
      torch.zeros(self.num_envs, dtype=torch.long, device=self.device),
    )
    self._a6_buffer[self._a6_cursor] = self._processed_actions
    target = self._a6_buffer[(self._a6_cursor - lag) % 6, self._a6_rows]
    self._a6_cursor = (self._a6_cursor + 1) % 6
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = target - encoder_bias
    unactuated = (self._env.episode_length_buf <= UNACTUATED_STEPS).unsqueeze(-1)
    if torch.any(unactuated):
      current = self._entity.data.joint_pos[:, self._target_ids]
      target = torch.where(unactuated, current, target)
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

"""Unitree G1 HoST standing-up task registration."""

from mjlab.tasks.registry import register_mjlab_task
from src.tasks.host_recovery.rl import HoSTOnPolicyRunner

from .env_cfgs import unitree_g1_host_standup_env_cfg
from .rl_cfg import unitree_g1_host_standup_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-HoST-StandUp",
  env_cfg=unitree_g1_host_standup_env_cfg(),
  play_env_cfg=unitree_g1_host_standup_env_cfg(play=True),
  rl_cfg=unitree_g1_host_standup_ppo_runner_cfg(),
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.a6 import a6_gminus_env_cfg, a6_gplus_env_cfg

for _arm, _env_cfg in (("GMinus", a6_gminus_env_cfg), ("GPlus", a6_gplus_env_cfg)):
  _a6_rl = unitree_g1_host_standup_ppo_runner_cfg()
  _a6_rl.experiment_name = f"g1_host_a6_{_arm.lower()}"
  _a6_rl.max_iterations = 10000
  _a6_rl.save_interval = 500
  _a6_rl.num_steps_per_env = 24
  register_mjlab_task(
    task_id=f"Unitree-G1-HoST-A6-{_arm}",
    env_cfg=_env_cfg(),
    play_env_cfg=_env_cfg(play=True),
    rl_cfg=_a6_rl,
    runner_cls=HoSTOnPolicyRunner,
  )

from src.tasks.host_recovery.pressed import pressed_env_cfg

_pressed_rl = unitree_g1_host_standup_ppo_runner_cfg()
_pressed_rl.experiment_name = "g1_host_pressed"
register_mjlab_task(
  task_id="Unitree-G1-HoST-Pressed",
  env_cfg=pressed_env_cfg(),
  play_env_cfg=pressed_env_cfg(play=True),
  rl_cfg=_pressed_rl,
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.supine_smp import supine_smp_env_cfg

_supine_smp_rl = unitree_g1_host_standup_ppo_runner_cfg()
_supine_smp_rl.experiment_name = "g1_host_supine_smp"
register_mjlab_task(
  task_id="Unitree-G1-HoST-SupineSmp",
  env_cfg=supine_smp_env_cfg(),
  play_env_cfg=supine_smp_env_cfg(play=True),
  rl_cfg=_supine_smp_rl,
  runner_cls=HoSTOnPolicyRunner,
)

from src.tasks.host_recovery.prone_smp import prone_smp_env_cfg

_prone_smp_rl = unitree_g1_host_standup_ppo_runner_cfg()
_prone_smp_rl.experiment_name = "g1_host_prone_smp"
register_mjlab_task(
  task_id="Unitree-G1-HoST-ProneSmp",
  env_cfg=prone_smp_env_cfg(),
  play_env_cfg=prone_smp_env_cfg(play=True),
  rl_cfg=_prone_smp_rl,
  runner_cls=HoSTOnPolicyRunner,
)

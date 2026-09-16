# HoST Recovery（人形站立恢复）任务迁移说明

本包把 [OpenRobotLab/HoST](https://github.com/OpenRobotLab/HoST)（RSS 2025,
*Learning Humanoid Standing-up Control across Diverse Postures*）的**恢复
（站立起身）框架**从 Isaac Gym + legged_gym 迁移到 mjlab 的 manager 架构下。

## 任务

| task id | 说明 |
| --- | --- |
| `Unitree-G1-HoST-StandUp` | 23 自由度 G1，平地，俯卧起始，学习自主站起 |

```bash
python scripts/list_envs.py --keyword HoST
python scripts/train.py Unitree-G1-HoST-StandUp --env.scene.num-envs=4096
python scripts/play.py Unitree-G1-HoST-StandUp --checkpoint-file <ckpt>
```

## 目录结构

```
src/tasks/host_recovery/
├── host_recovery_env_cfg.py   # 任务基配置（奖励/观测/事件/终止/课程/度量）
├── config/g1/
│   ├── env_cfgs.py            # G1 专属：刚体名、关节名、动作缩放、play 覆盖
│   ├── rl_cfg.py              # PPO 超参（取自 HoST G1CfgPPO）
│   └── __init__.py            # 任务注册
├── mdp/
│   ├── rewards.py             # task/regu/style/target 四组奖励
│   ├── observations.py        # 76 维单步观测 + 历史
│   ├── events.py              # 俯卧/仰卧重置、关节扰动
│   ├── terminations.py        # 速度越界终止（无接触终止）
│   ├── curriculums.py         # action_scale 训练轮
│   ├── pull_force.py          # 垂直拉力训练轮
│   └── metrics.py             # 头部高度进度 + 共享状态
└── rl/runner.py               # ONNX 导出的 runner
```

## 核心机制对照

| HoST（Isaac Gym） | 本包（mjlab） |
| --- | --- |
| `compute_reward` 里 task 组相乘 | `mdp.standup`：`task_orientation * task_head_height * 2.5` |
| 四个 reward group 权重 `[2.5, 0.1, 1, 1]` | 权重折算进各 term：`scale * group_weight * HOST_CONSTRAINT_DT` |
| 约束项统一 `* dt`（`_prepare_reward_function`） | `mdp.HOST_CONSTRAINT_DT = 0.02` |
| 阶段门控 `target_base_height_phase1/3` | `phase1_height=0.45` / `phase3_height=0.65` 参数 |
| `_get_noise_scale_vec` 逐通道噪声 | `mdp.host_noise_vector` + `host_observation` 内部加噪 |
| `compute_observations` 整体乘 unactuated 掩码 | `host_observation` 内 `episode_length_buf > 120` 掩码 |
| 历史 6 帧 | `ObservationGroupCfg(history_length=6, history_ordering="time")` |
| `_reset_dofs`：`default * U(0.9,1.1) + U(-0.1,0.1)` | `mdp.reset_joints_scaled` |
| `init_state` 俯卧位姿 `[0,0,0.5]` + 180° about y | `mdp.reset_root_state_posture(posture="prone")` |
| `update_force_curriculum` 衰减 action_rescale | `mdp.action_scale_decay` |
| 拉力课程 `force -= 20` | `mdp.pull_force_decay`（默认关闭，见下） |
| `terminate_after_contacts_on = []` | 不注册任何接触终止项 |
| `dof_vel_out` / `base_vel_out` | `mdp.dof_velocity_out_of_bounds` / `base_velocity_out_of_bounds` |

## 已知差异 / 未迁移项（务必阅读）

**会影响行为，需要你决策：**

1. **动作不是增量式**。HoST 是 `target = dof_pos + action * action_rescale`（每步以
   实测位置为基准的增量），mjlab 的 `JointPositionAction` 是
   `default_joint_pos + scale * action`。两者不等价，本包沿用了仓库其它任务的
   `use_default_offset=True`。
2. **没有 head 刚体**。HoST 用 URDF 里的 `keyframe_head` 虚拟链接测头高，mjlab 的
   G1 MJCF 没有 head 刚体（头是焊在 `torso_link` 上的网格），因此站立奖励与进度
   度量改成「躯干高于脚」。躯干比 HoST 的 head keyframe 低约 0.2 m，
   `target_head_height=1.0` 可能需要下调才可达。
3. **拉力训练轮默认关闭**（`pull_force.force = 0`）。施加外力的逐步钩子在本地无法
   验证（`mjlab==1.2.0` 是 Linux-only 的 pip 依赖，本机未安装），
   `mdp/pull_force.py` 里的 `apply_pull_force` 会在运行时探测可用的 wrench 写入
   接口，找不到就抛出说明性错误。确认接口后再打开。
4. **奖励实现不完整**。以下 HoST 奖励项没有迁移，配置里以注释保留了原始权重：
   `regu_smoothness`（需要二阶动作历史）、`regu_torques`、
   `regu_joint_power`（需要施加力矩字段）、`regu_joint_tracking_error`（需要 PD 目标）、
   `regu_dof_vel_limits`（需要软关节速度上限）。
5. **多 critic 与平滑正则未迁移**。HoST 的 PPO 有四组独立 critic（优势按组权重加权
   求和）和 policy/value smoothness 正则，这在 mjlab 的 `RslRlPpoAlgorithmCfg` 里
   没有对应项，需要改 vendored `rsl_rl`。当前是单 critic 的加权求和奖励。
6. **域随机化只迁移了部分**。已迁移：脚部摩擦、编码器偏置、质心偏移。未迁移：
   kp/kd 随机化、电机强度、负载质量、连杆质量、延迟（`max_delay_timesteps`）、
   外力推动 —— mjlab 的 manager API 未在本仓库中提供已验证的对应项。
7. **critic 观测是干净的**。HoST 把同一个带噪观测同时喂给 actor 和 critic；mjlab 的
   两个观测组独立计算，这里让 critic 用无噪版本（与仓库其它任务一致）。

**忠实保留、未做"修正"的地方：**

8. `style_ground_parallel` 按 HoST 原样移植：它对**单个**踝部刚体求 `var(dim=1)`，
   在默认 `correction=1` 下是 NaN，随后的 `var < 0.05` 恒为 False，因此该项恒为
   常数 0。仍然保留在奖励表里，以便与 HoST 配置逐项对应。代码上已补回
   `[body_id]` 单例维度，避免索引结果降维后 `.var(dim=1)` 直接崩溃。
9. `style_hip_yaw_deviation` / `style_hip_roll_deviation` / `style_knee_deviation`
   的判据写法（`max(|q|) > a or min(|q|) > b` 这类组合）照抄 HoST 原样。

## 本次修正（sjw）

- `style_ground_parallel` 保留 `[body_id]` 单例维度，修复训练时崩溃。
- `HoSTMetrics` 在 reset 时先保存 `last_episode_head_height`，课程衰减改读该
  快照，修复 reset 清空后 `action_scale` 永远无法下降的问题。
- 观测表格把 “previous action” 改为 “current action”，与 HoST 源码一致。

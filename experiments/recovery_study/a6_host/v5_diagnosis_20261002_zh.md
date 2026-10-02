# V5 检查和训练记录（2026-10-02）

执行代码提交：`7157543`，分支 `codex/a6-host-v5-fidelity`。
冻结 A6 参考提交：`1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`。

## 已完成的修正

头部点位按 torso 的世界旋转重建，并在速度中计入角速度叉乘局部偏移。
关节成本使用 joint/DoF 顺序的 qfrc_actuator。上升 overspeed 使用历史
relaxed 口径。completion 使用冻结代码的 signed separation 和 above-board
豁免，G 保持原二维投影。两个板场景均要求真实曾接触，并保留 25 步
未接触 invalid 保护。评估新增逐 trial invalid 类型和首次发生时间。

双组 32 环境预检通过，包含 510 控制步、dt/奖励映射、部分 reset 隔离、
头部旋转点位/速度、垂直超速和 completion/G 几何差异反例。
4096 环境、30 更新的双组短训练完成；dof_vel_out/base_vel_out 为零，
loss 有限，actor 来源 SHA 和 fresh critic SHA 配对一致。

## 进一步诊断

使用修正检查器重测 V4 final：每组 1024 环境、20 秒、seed 20261021。
G−/G+ Clear 为 26.56%/28.52%，严格 SR1/SR10 与 height/upright core SR
均为零。此前固定世界 Z 偏移高估头高，V4 旧 core SR 不能继续用于说明
站立成功；旧几何口径 Clear 也不能和修正后结果合并。

导向板 invalid 为 57.03%/64.06%，主要原因是超过 1500 N 板力保护；
未接触原因均为零。自由板 invalid 为 1.95%/2.73%，主要由穿透保护触发。
保护阈值和冻结奖励权重保持原值。

原生 HoST29 checkpoint 与短训练模型各做 128 环境、20 秒历史 validation：
Native / G−短训练 / G+短训练 Clear 为 42.19% / 35.94% / 39.06%；
三者严格和 core SR 均为零。该小样本用于接口与数值检查，不能用于
选最终模型或证明稠密引导有效。原生策略本身也尚未满足 A6 稳定站立。

## 正式训练

远端工作目录：`/home/sjw/AMP_mjlab-a6`。
输出：`logs/a6_host_adaptation_20261002_v5/`。
GPU0：G−，PID `468827`；GPU1：G+，PID `468828`。
两组从 `artifacts/host29_recovery_20261001/model_10000.pt` 重启；
不使用短训练优化后的权重。seed 20261013，4096 环境×24 步×10000 更新，
每 500 更新保存。learning rate 1e-4，std 0.8 冻结，actor normalizer 冻结，
critic/critic normalizer/optimizer 重新初始化。原生辅助课程保留。

短训练约 5.3 秒/更新，正式完成时间估算约 15 小时，受运行负载影响。
后续首先在 500 更新 checkpoint 检查真实头高、膝角、脚宽、静止门、
板力保护率和四方向能力，再按同一验证口径检查是否收敛。
目前仅一个正式配对训练 seed；历史 validation 不作为独立最终 test。

# V5 historical A6 fidelity correction

V4 evaluation identified zero strict SR1/SR10 and high guided-plate invalidity.
Comparison against SMP commit `1b6d7e61ddbcd2caf9ea17b02160dd1cf0d119d2`
found runtime discrepancies that must be corrected before interpreting training.

- Reconstruct the torso-local head point using its rotated 0.43 m offset;
  include angular velocity crossed with that offset in point velocity.
- Use joint-ordered generalized actuator torque for joint costs and power.
- Match relaxed A6 upward overspeed limits: 0.30/0.30/0.30/0.20 m/s,
  retaining the 0.20 m/s downward limit.
- Completion uses signed maximum-axis XY separation and the historical
  above-board exemption. G retains planar overlap without that exemption.
- Both board types start with `ever_contact=False`; both use the historical
  no-contact invalid guard after 25 control steps.
- Add per-trial invalid cause flags and first-invalid time to evaluation.

V5 starts both arms from the published native HoST29 checkpoint, not V4.
Retain V4 optimizer settings for this correction experiment: learning rate
1e-4, frozen exploration standard deviation 0.8, frozen actor normalizer,
fresh common critic/optimizer. Preserve reward weights, randomization and
native HoST curricula. Run a short paired 4096-environment training first,
then restart from the native checkpoint for the full 10000-update budget.

Historical validation banks remain validation, not independent final tests.
V4 results used the old geometry/head semantics and cannot be pooled with V5.
Low strict standing success needs further measured diagnosis after correction;
this change does not add posture or position rewards forbidden by the protocol.

# HoST constrained recovery evidence

This supplement documents my HoST work on a 29 DoF G1 in MuJoCo/mjlab, extending standing-up control to sustained overhead plate constraints. It covers control adaptation, staged escape rewards, target-critic adaptation, evaluation and debugging. Original HoST multi-critic PPO, curricula and motion constraints are upstream methods.

## Method

- Adapt 23 to 29 DoF: align observations, actions, inertia, joint limits and waist/wrist torque limits; use posture targets and upper-body residuals to address persistent joint-limit contact.
- Before escape, reduce selected upward recovery incentives. Reward torso progress toward the plate exit, supported movement and arm withdrawal. Knee/hip penalties and extension/release gates discourage reducing coverage through leg curling alone.
- Reuse the existing target critic for escape progress. Task, regularisation, style and target groups estimate returns separately. During entrapment, task/style advantages are scaled by 0.05 after normalisation; reducing raw rewards alone can be cancelled by advantage normalisation.
- Confirm escape after prior plate contact and 25 consecutive control steps with no contact and clearance. The presentation specifies 4 cm horizontal clearance for a guided plate and 2.5 cm geometric clearance for a free plate. Restore recovery emphasis using the same actor.
- Keep action-cost and safety terms active. Fixed teachers provide training references; the student actor and critics continue to learn.

## Final simulation summary

| Scene | Standing-up success |
| --- | ---: |
| Vertically guided plate | 87.98% |
| Free rigid plate | 98.01% |

These values are reproduced exactly from data/host-results.csv. The two-row final summary does not contain trial counts, checkpoint IDs, seeds or confidence intervals. Do not reconstruct counts, pool it with older tables, or present it as a controlled baseline ranking. The separate older flat-posture CSV does not supply denominators for these final scene rates.

Standing-up success, geometric escape and sustained standing are distinct quantities. The historical presentation protocol confirms escape within a rollout and checks the final valid standing state, without necessarily requiring continuous stable standing or renewed final clearance. Preserve the protocol of a later full evaluation before claiming a match.

## Demonstration videos

- [Four-scene overview](media/HoST_model_02999_seed42_four_scenes.mp4)
- [Guided plate prone start](media/HoST_model_02999_seed42_guided_prone.mp4)
- [Free plate supine start](media/HoST_model_02999_seed42_free_supine.mp4)

These selected successful simulation trajectories are labelled v24 model_02999 seed 42 in the asset names. They illustrate behaviour, not random-sample success rates. They are not assigned to the final summary without a matching evaluation manifest. HoST has no hardware verification for this task.

## Scope and attribution

The reward/critic changes were developed with curriculum, teacher and control adjustments. No isolated ablation establishes the benefit of a single change. HoST style constraints are not an AMP adversarial discriminator. HoST and AMP actor information conditions differ and require control in future common-protocol comparisons.

This is a documentation/evidence update on the tholin_host snapshot 7afd8ae. It does not change or validate new training code. The earlier V38/V40 flat height audit is a separate diagnostic and must not be merged with these constrained-recovery rates.

The local Interview presentation and associated CSV/video snapshot were reviewed on 6 November 2026. source_host_excerpt.txt preserves the HoST text; SHA256.json records exact packaged bytes.

## Upstream method

Huang et al., Learning Humanoid Standing-up Control across Diverse Postures, RSS 2025. Official project: https://github.com/InternRobotics/HoST

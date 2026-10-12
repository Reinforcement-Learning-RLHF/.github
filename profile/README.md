# Preference Learning for Robot Manipulation (WAT.ai, 2026)

Teaching an SO-101 robot arm to pour liquid by asking "which attempt looked better?" instead of collecting more expert demonstrations. The RLHF recipe from [Christiano et al. 2017](https://arxiv.org/abs/1706.03741), applied to physical manipulation where every attempt costs real seconds on real hardware.

This folder archives the January–September 2026 work: code, data pointers, the project presentation, and a full technical report.

## The idea

Imitation learning is slow (roughly 1 hour of hand-driving per hour gained) and the policy can never beat its teacher. Instead, we compare pairs of attempts (A vs B) and train a reward model to score them. Three independent judges produce the preference labels, with everything downstream held identical so differences in the trained robot come only from who judged:

1. **Scripted**: a rule computes the label from simulator ground truth (amount poured, spill, tilt smoothness). Sim only.
2. **Human**: a person watches both clips and clicks.
3. **VLM**: a vision-language model (Gemini) is shown the videos and asked. The only automatic judge that runs on hardware.

Pouring was chosen because it gives graded outcomes, where "how well" matters more than "did it happen". Stirring was dropped as too ambiguous for labellers to agree on.

## What exists (Sept 2026)

- **51 teleoperated SO-101 pouring demos** on real hardware: 53,409 frames at 30 fps, 6 joints of state/action plus a 640x480 front camera. Published at [huggingface.co/datasets/ammiellewb/wat.ai-so101](https://huggingface.co/datasets/ammiellewb/wat.ai-so101). Demos average 34.9 s with consistent structure (grip at 10–13 s, pour, release), which is what makes them learnable by ACT.
- **A trained ACT policy**: 50k-step checkpoint, loss 0.050, trained on WATCloud SLURM (RTX 3090).
- **Preference-labelling pipeline** working end to end for all 3 sources, plus VLM scoring runs on 42 sim rollouts and the 51 hardware demos.
- **The blocker**: 1,540 scripted pairs were generated, but 86.5% are ties because 52 of 56 registry episodes are successes. Preferences need bad attempts to compare against good ones. Diversity has to be injected via policy rollouts (noise, moved cups), not more clean demos.

Reward-model training itself is scaffolded but not yet built. Next steps were: collect mixed-quality rollouts, gather 150–300 labels per judge, train 3 reward models, retrain the policy on each, and measure whether pours improve. Target venue: CUCAI 2027. A negative result (VLM labels too noisy to learn a reliable reward) counts as a result.

## Contents

```
WAT.ai Preference Feedback for Robot Manipulation.pdf   16-slide project presentation
PROJECT_REPORT_2026.md                                  full technical report, every number cited to a file
robot-pref-learning-project-hub/   main repo: labelling (scripted/human/VLM), reward-model scaffold, Streamlit app
so101-environment/                 MuJoCo SO-101 pouring sim, scripted demo collection, LeRobot Hub I/O
so101-teleop-demo/                 hardware run-book (power, calibration, teleop command) + session notes
machine-learning-repo/             early VLM demo-evaluation pipeline and trajectory-drawing experiments
```

Code lives under [github.com/Reinforcement-Learning-RLHF](https://github.com/Reinforcement-Learning-RLHF). The ACT training toolchain was upstream [huggingface/lerobot](https://github.com/huggingface/lerobot) v0.5.1 (not archived here).

## Engineering lessons (from the presentation)

- Consistency beats volume: one strategy recorded the same way every time; mixed styles average into hesitation.
- Ties mean no signal: coarse yes/no metrics make a judge call most pairs equal.
- Scripted labels stay in sim; on hardware only the VLM runs without a person.
- Infrastructure is the work: all 3 judges answer in the same format so the comparison is fair.

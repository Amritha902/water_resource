# Two base papers: 4 lines each

## Paper 1: MAACC (Wang et al., IEEE T-II 2026)
- **Their result:** two RL agents on top of a PID cut oxygen tracking error by about 95% and nitrate error by about 74% versus PID, still 86% / 69% with actuator noise, and over 90% of runs succeed.
- **We obtained:** run exactly as printed, with all their Table I values, on the official BSM1 plant and influent, MAACC equals PID in all 180 runs; its learned correction never grows. The plant itself checks out (our PID is within ~30% of theirs). So their table cannot be reproduced from the paper alone.
- **Mechanism followed:** yes, exactly: equations (7)–(19) and (27), Table I, raw units, 20 runs per case.
- **Our modified version (not theirs):** bigger steps, scaled inputs and a safety fallback to PID reach oxygen 81% / 75% below PID in dry / rain, 18 of 20 runs better than PID, but it is worse than PID under noise.

## Paper 2: floating DO (Du et al., Sci China Tech Sci 2023)
- **Their result:** letting dissolved oxygen float, instead of holding 2 mg/L, saves 5–7% aeration energy with effluent still within limits.
- **We obtained:** 6.8% / 4.9% / 5.0% on dry / rain / storm (paper 7.24 / 5.52 / 6.06). But the ammonia limit is broken 35–38% of the time, against 12–23% for PID.
- **Mechanism followed:** yes. DDPG, same state, rewards, network sizes and hyperparameters, train on week 1, test on week 2.
- **Changes:** as published it collapses to zero air. It needed the air setting added to the state, a 15-minute decision step, rewards scaled by (1 − γ), and a penalty that stops the actor saturating.

Both: our plant and influent are rebuilt from the model definition, so results match in direction and size, not exactly.

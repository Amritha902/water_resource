# Two base papers: 4 lines each

## Paper 1: MAACC (Wang et al., IEEE T-II 2026)
- **Their result:** two RL agents on top of a PID cut oxygen tracking error by about 95% and nitrate error by about 74% versus PID, still 86% / 69% with actuator noise, and over 90% of runs succeed.
- **We obtained:** run exactly as printed, with all their Table I values, on the official BSM1 plant and influent, MAACC equals PID in all 180 runs; its learned correction never grows. The plant itself checks out (our PID is within ~30% of theirs). So their table cannot be reproduced from the paper alone.
- **Mechanism followed:** yes, exactly: equations (7)–(19) and (27), Table I, raw units, 20 runs per case.

## Paper 2: floating DO (Du et al., Sci China Tech Sci 2023)
- **Their result:** letting dissolved oxygen float, instead of holding 2 mg/L, saves 7.24% (dry) and 5.52% (rain) aeration energy versus PID, effluent still within limits.
- **We obtained:** on the official BSM1 plant the PID and fuzzy comparators match their tables within 1%. Our DDPG-B saves 10.6% (dry) and 7.1% (rain) on average, with more ammonia than theirs; the 3 seeds spread from 5.8% to 15.7%, and one seed lands on their numbers (3513 vs 3430 kWh/d, DO 1.01 vs 1.03).
- **Mechanism followed:** yes: DDPG, same state, rewards, network sizes and hyperparameters, train on week 1, test on week 2.
- **Changes:** we added the air setting to the state, a 15-minute decision step and rewards scaled by (1 − γ), because on our first (rebuilt) plant the published version collapsed to zero air. On the official plant the published version also ends near a 10.7% saving (1 seed), after shutting the air almost off mid-training. Not reported by them: the ammonia limit is broken far more often than under PID.

Both: run on the official BSM1 plant and influent (`bsm2-python`). The official storm file was not available, so storm is only on our rebuilt plant.

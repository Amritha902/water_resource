# Two base papers: 4 lines each

## Paper 1: MAACC (Wang et al., IEEE T-II 2026)
- **Their result:** two RL agents on top of a PID track oxygen and nitrate more tightly than other methods, stay robust to actuator noise, and train successfully over 90% of the time.
- **We obtained:** under their noise protocol, nitrate tracking beats PID by +27% (dry), +21% (rain), +10% (storm). Clean runs are unchanged from PID. Part of the gain is what an anti-windup PID also gives. Not checked: their table against seven other methods.
- **Mechanism followed:** yes. Two agents, PID plus learned correction starting at zero, incremental control, cost on both errors.
- **Changes:** their critic update (eq. 15) diverges, so we use TD(0). Plus a bias on the critic, normalised inputs, a clipped correction, and our own rebuilt plant.

## Paper 2: floating DO (Du et al., Sci China Tech Sci 2023)
- **Their result:** letting dissolved oxygen float, instead of holding 2 mg/L, saves 5–7% aeration energy with effluent still within limits.
- **We obtained:** 6.8% / 4.9% / 5.0% on dry / rain / storm (paper 7.24 / 5.52 / 6.06). But the ammonia limit is broken 35–38% of the time, against 12–23% for PID.
- **Mechanism followed:** yes. DDPG, same state, rewards, network sizes and hyperparameters, train on week 1, test on week 2.
- **Changes:** as published it collapses to zero air. It needed the air setting added to the state, a 15-minute decision step, rewards scaled by (1 − γ), and a penalty that stops the actor saturating.

Both: our plant and influent are rebuilt from the model definition, so results match in direction and size, not exactly.

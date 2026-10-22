# Two-minute talk for faculty

Good morning. My work is on control of a wastewater treatment plant using
reinforcement learning.

I ran both base papers on the official BSM1 benchmark plant, the same
simulator the papers use, with its official influent data. I followed each
paper exactly as printed first, and only then changed anything.

**Paper 1, by Wang and co-authors,** puts two learning agents on top of a PID
controller: one holds oxygen, one holds nitrate. The paper reports errors about
95% lower than PID for oxygen and about 74% lower for nitrate.

I implemented it exactly as printed, with every value from their Table I, on
the official benchmark plant and influent. The plant checks out: my PID is
within about 30% of their PID numbers. But the algorithm as printed does not
learn. In all 180 runs its correction stays near zero and the result equals
the PID. So their Table II cannot be reproduced from the paper alone; it
depends on details they do not give.

To see what the idea can do, I also built a modified version, with bigger
steps, scaled inputs and a safety fallback to the PID. It reaches 81% lower
oxygen error in dry weather and beats PID in 18 of 20 runs. But that is my
variant, not their algorithm, and under noise it is worse than PID.

**Paper 2, by Du and co-authors,** lets oxygen float to save aeration energy.
They report 5 to 7% saving. I get 6.8, 4.9 and 5.0% for dry, rain and storm,
so this one reproduces. But I also found something the paper does not report:
the saving comes with the ammonia limit broken about three times as often as
under PID.

**My contribution** is a better agent for paper 2. It adds an influent
forecast and a risk-aware critic, and it handles the limits as constraints.
At the same violation rate, it uses about 29 kWh per day less than paper 2's
agent.

All numbers come from saved runs in the repository, and the limits are written
down next to them. Thank you.

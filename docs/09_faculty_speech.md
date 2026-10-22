# Two-minute talk for faculty

Good morning. My work is on control of a wastewater treatment plant using
reinforcement learning.

I ran both base papers on the official BSM1 benchmark plant, the simulator
the papers use, with its official influent data. I followed each paper
exactly as printed first, and only then changed anything.

**Paper 1, by Wang and co-authors,** puts two learning agents on top of a PID
controller: one holds oxygen, one holds nitrate. They report errors about 95%
lower than PID for oxygen and about 74% lower for nitrate.

I implemented it exactly as printed, with every value from their Table I, and
ran 20 runs per case, as they did. The plant checks out: my PID is within
about 30% of their PID numbers. But the algorithm as printed does not learn.
In all 180 runs its correction stays near zero, and the result equals the PID.
I also tried twelve different starting weights, the one thing the paper leaves
open; none reproduces their table. So their results cannot be obtained from
the paper alone.

To see what the idea can do, I built a modified version, with bigger steps,
scaled inputs and a safety fallback to the PID. It cuts oxygen error by 81% in
dry weather and beats PID in 18 of 20 runs. But that is my variant, not their
algorithm, and under noise it is worse than PID.

**Paper 2, by Du and co-authors,** lets oxygen float to save aeration energy.
They report 7.2% saving in dry weather and 5.5% in rain. On the official plant
my PID and fuzzy comparators match their numbers within 1%. My DDPG agent
saves 10.6% and 7.1% on average over three training seeds, and one seed lands
almost exactly on their result. But the seeds spread widely, and as published
the agent shuts the air almost off and ammonia goes far over the limit: it
needed three fixes before it learned. I also found what the paper does
not report: with that saving, effluent ammonia is above its limit 56% of the
week, against 21% under PID.

**My contribution** is a better agent for paper 2. It adds an influent
forecast and a risk-aware critic, and it treats the limits as constraints. On
our rebuilt plant, at the same violation rate, it uses about 29 kWh per day
less than paper 2's agent.

In short: paper 2 reproduces in direction, paper 1 does not reproduce as
printed, and every number I quoted comes from saved runs in the repository.
Thank you.

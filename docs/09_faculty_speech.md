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

**Paper 2, by Du and co-authors,** lets oxygen float to save aeration energy.
They report 7.2% saving in dry weather and 5.5% in rain. On the official plant
my PID and fuzzy comparators match their numbers within 1%. My DDPG agent
saves 10.6% and 7.1% on average over three training seeds, and one seed lands
almost exactly on their result. But the seeds spread widely. Run exactly as
published, it shuts the air almost off during training, then recovers to a
similar 10.7% saving. I also found what the paper does
not report: with that saving, effluent ammonia is above its limit 56% of the
week, against 21% under PID.

In short: paper 2 reproduces in direction, paper 1 does not reproduce as
printed, and every number I quoted comes from saved runs in the repository.
Thank you.

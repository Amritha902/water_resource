# Two-minute talk for faculty

Good morning. My work is on control of a wastewater treatment plant using
reinforcement learning.

I first rebuilt the standard benchmark plant, BSM1, in code, so every method
runs on the same simulated plant. Then I reimplemented my two base papers on it.

**Paper 1, by Wang and co-authors,** puts two learning agents on top of a PID
controller: one holds oxygen, one holds nitrate. The paper reports errors about
95% lower than PID for oxygen and about 74% lower for nitrate.

I ran their full protocol, 20 runs per case. In dry and rain weather, oxygen
error comes out 81 and 75% below PID. Nitrate in a typical run is 84% below
PID. 18 of 20 runs beat PID, against their "over 90%". In storm, oxygen still
improves, nitrate does not. Their noise experiment I could not reproduce: under
the stated noise even my PID gets three times worse, while theirs barely
changes, so their noise must reach the plant differently.

To get there I had to fix two things. Their critic update, equation 15,
diverges, so I use the standard TD form. And I added a safety fallback: if
learning makes tracking worse, control goes back to the PID smoothly. Without
it, 8 of 20 runs failed.

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

"""Highlight both base papers and attach our reproduction notes.

The PDFs are not in the repository (copyright). With your own copies:

    python tools/annotate_papers.py paper1.pdf paper2.pdf out1.pdf out2.pdf
"""
import sys, json, pymupdf

Y, G, R, B = (1, 0.92, 0.3), (0.6, 0.93, 0.6), (1, 0.6, 0.6), (0.6, 0.8, 1)
WHO = "Amritha - reproduction notes"
FI, FFL = "ﬁ", "ﬄ"   # ligatures as extracted

P1 = [
 (1, "multiple agents are created,", Y, "Core idea: one agent per loop. Agent 1: KLa5 -> S_O,5; agent 2: Qa -> S_NO,2. Code: src/wwtp/maacc/agent.py."),
 (1, "the prior control strategy with expert", Y, "u = a + b: learned correction a on top of the PID b. a starts at 0, so day one is plain PID (checked by test_maacc_starts_from_the_expert_prior)."),
 (1, "tal control strategy, which improves the anti-interference", Y, "Claim 1: incremental control -> robust to disturbances. Tested by Table III (noise): NOT reproduced in our runs."),
 (1, "The stability of the", B, "Theorem 1 (UUB). We did not re-prove it. Note: UUB allows a large bound, so it does not rule out the eq. (15) blow-up we found (p. 4)."),
 (2, "suffer from low training success rates", Y, "Motivation: other RL controllers train unreliably. Their claim: MAACC succeeds > 90 % (p. 9)."),
 (3, "improved into the following incremental control strategy", Y, "Eq. (8): Delta a = pi(z), a_k = a_{k-1} + Delta a. We keep this exactly: no actor bias, pure integration (an actor bias made a drift into its clip - fixed)."),
 (3, "Wi, Yi, Ri, and Pi are positive", B, "Eq. (10) weights. Values not in the text. Ours (tuned): W=30, P=1, Y=0.02, R=0, inputs normalised by actuator range."),
 (4, "approximation error of the CN is expressed as", R, "PROBLEM FOUND: eq. (15)+(18) as printed diverge. For slowly varying states Q <- Q(1 + lc*lambda*(1-lambda)*||theta||^2) - ..., multiplier > 1 for any lc > 0. Reproduced in tests/test_critic_update.py. We use the standard TD(0) direction instead."),
 (4, "the loss function of the AN is", Y, "Eq. (17): actor minimises 1/2 Q^2. Since a_k = a_{k-1} + Delta a_k, both critic inputs depend on the actor: we use dQ/dDelta a + dQ/da."),
 (4, "values (ranging from 0.001 to 0.01)", B, "Learning rates. Ours: l_a = 0.01, l_c = 0.05. l_c above their range was needed for the critic to keep up (20-seed search)."),
 (5, "of the CN and the AN for the ith agent are UUB if the following", B, "UUB bounds the weight error but allows a large bound - consistent with the divergence we see for eq. (15)."),
 (7, "set to 45 s, and the total simulation duration is set to 14 days.", G, "Same protocol in our runs: 45 s, 14 days, first day excluded (PID warm-up)."),
 (7, "based on industry experience.", G, "Set points r = [1, 2] used as given."),
 (7, "diag{100, 100 000}", G, "Same PID gains (src/wwtp/baselines/pid.py). Our PID IAE S_O,5 = 0.0076 (paper 0.0114)."),
 (8, "Trajectories of SO,5 and SNO,2 under the dry weather.", G, "Our version of this figure: docs/figures/readme_paper1_tracking.png - MAACC stays much closer to both set points than PID."),
 (8, "white noise with a mean of zero and a standard deviation of 3%", R, "NOT REPRODUCED. With this noise our PID gets 3x worse (IAE 0.0076 -> 0.0243); the paper's PID only +10 % (0.0114 -> 0.0125). Their noise must reach the plant differently. Our MAACC is worse than PID here."),
 (9, "TABLE II", G, "OURS vs PAPER (mean of 20 runs, error reduction vs PID). Oxygen: dry -81 % (paper -95), rain -75 % (-95), storm -32 % (-92). Nitrate: dry -22 % mean / -84 % median (paper -74), rain +32 % mean / -83 % median (-71), storm no gain (-76). Absolute values differ: rebuilt plant and influent."),
 (9, "TABLE III", R, "Noise case: paper -86 % (O2), -69 % (NO3). Ours: worse than PID (+128 % / +198 %). 0/20 runs beat PID."),
 (9, "and III are the average values obtained from 20 simulation", G, "We also ran 20 runs per case: 360 runs in total (results/summary_paper1_full_*.json)."),
 (9, "training is deemed successful", G, "Same success definition used. Ours: 18/20 dry, 18/20 rain, 4/20 storm, 0/20 noise."),
 (9, "the MAACC algorithm can reach over 90% across 20 simulation", Y, "As printed we got 12/20: 8 runs diverged in the first hour (untrained critic slope sends the actor to its rail). With our safety fallback (hand back to PID, bumpless) 18/20."),
]

P2 = [
 (1, "reward functions and by analysing the relationships among", Y, "Two rewards: A (effluent limits, eq. 19) and B (energy + ammonia, eq. 23). Code: src/wwtp/rl/rewards.py."),
 (7, "this paper selects S NH, S s, and S O as the input of the agent,", R, "PROBLEM FOUND: with an incremental action the actuator KLa5 is a hidden state - not Markov. As published the agent collapsed to zero air. We add KLa5/240 to the state."),
 (7, "Figure 4 shows the variation of e" + FFL + "uent parameters when", G, "Reproduced: NH4 falls and Ntot rises with DO, COD/BOD/TSS barely move, energy rises (results/summary_fig4.json)."),
 (8, "environment at an interval of 45 s and obtain the required DO", R, "45 s steps with gamma = 0.99 give a ~75 min horizon, but effluent NH4 responds over hours. We hold each action 15 min."),
 (8, "the in\ufb02uent data are taken as the in\ufb02uent of WWTPs in the", G, "Same protocol: learn on week 1, evaluate on week 2."),
 (9, "the size of the replay bu" + "ﬀ" + "er is set to R = 30000;", B, "Same hyper-parameters: gamma 0.99, batch 256, tau 0.001, lr 1e-4 / 3e-4, actor/critic 256-256. Plus reward scaled by (1 - gamma) - the critic's dQ/da had the wrong sign for weeks without it."),
 (9, "ﬁnally we let β1 = 1, β2 = 0.42.", B, "beta2 = 0.42 used as given. AE entered as KLa5/240 (units not stated; raw AE ~3500 would swamp the 0.42 term)."),
 (10, "AE is reduced by 5.52% and", G, "REPRODUCED. Rain saving ours 4.9 % (paper 5.52 %)."),
 (10, "AE is decreased by 7.24% and 7.22%", G, "REPRODUCED. Dry saving ours 6.8 % (paper 7.24 %). PID AE dry ours 3719.7 kWh/d vs paper 3698.2 (+0.6 %)."),
 (10, "6.06% and 6.03%, respectively in", G, "REPRODUCED. Storm saving ours 5.0 % (paper 6.06 %)."),
 (10, "appears to increase under DDPG-A and DDPG-B", R, "NOT REPORTED BY THE PAPER: how often the 4 g N/m3 ammonia limit is broken. Ours: DDPG-B 38 % of the week (dry) vs 12 % for PID. The energy saving is bought with violations - this motivates our constrained, risk-aware agent."),
 (10, "Comparisons of AE and EQ under four di" + "ﬀ" + "erent control strategies", Y, "Mean S_O,5 dry: PID 1.998 (paper 2.000), DDPG-B 1.130 (1.032), DDPG-A 1.894 (1.415) - DDPG-A not matched, ours settles more conservatively."),
]

def run(src, dst, items, summary):
    doc = pymupdf.open(src)
    miss = []
    for page_no, phrase, col, note in items:
        page = doc[page_no - 1]
        quads = page.search_for(phrase, quads=True)
        if not quads:
            # try the neighbouring pages (page numbering of the extract may be off by one)
            for alt in (page_no, page_no - 2):
                if 0 <= alt < len(doc) and (q := doc[alt].search_for(phrase, quads=True)):
                    page, quads = doc[alt], q
                    break
        if not quads:
            miss.append(phrase)
            continue
        h = page.add_highlight_annot(quads[:1] if len(quads) > 1 else quads)
        h.set_colors(stroke=col)
        h.set_info(title=WHO, content=note)
        h.update()
        r = quads[0].rect
        x = page.rect.width - 22 if r.x0 > page.rect.width / 2 else 6
        t = page.add_text_annot(pymupdf.Point(x, r.y0), note, icon="Comment")
        t.set_info(title=WHO)
        t.set_colors(stroke=col)
        t.update()
    p0 = doc[0]
    s = p0.add_text_annot(pymupdf.Point(p0.rect.width - 30, 20), summary, icon="Note")
    s.set_info(title=WHO); s.set_colors(stroke=(1, 0.5, 0)); s.update()
    doc.save(dst, garbage=3, deflate=True)
    print(dst, "missing:", miss)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[3], P1,
        "SUMMARY - Paper 1 reproduced on our rebuilt BSM1, 20 runs per case. Dry and rain: yes (oxygen error -81/-75 % vs PID, nitrate median -84/-83 %, 18/20 runs beat PID). Storm: oxygen yes, nitrate no. Noise (Table III): no. Changes needed: TD(0) critic (eq. 15 diverges), larger steps, safety fallback to PID. Legend: green = matched, red = problem / not reproduced, yellow = key claim, blue = parameter.")
    run(sys.argv[2], sys.argv[4], P2,
        "SUMMARY - Paper 2 reproduced: energy saving 6.8 / 4.9 / 5.0 % (paper 7.24 / 5.52 / 6.06). Needed: KLa5 in the state, 15-min decisions, reward scaled by (1 - gamma). Not reported by the paper: the ammonia limit is broken ~3x as often as under PID. Legend: green = matched, red = problem / not reported, yellow = key claim, blue = parameter.")

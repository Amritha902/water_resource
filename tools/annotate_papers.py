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
 (3, "Wi, Yi, Ri, and Pi are positive", B, "Eq. (10) weights - given in Table I (p. 7): W=(1,2), Y=(0.5,0.05), R=(1e-4,1e-8), P=(2I, I). We missed that table at first (it is an image). The exact replication (src/wwtp/maacc/exact.py) uses these values."),
 (4, "approximation error of the CN is expressed as", R, "PROBLEM FOUND: eq. (15)+(18) as printed diverge. For slowly varying states Q <- Q(1 + lc*lambda*(1-lambda)*||theta||^2) - ..., multiplier > 1 for any lc > 0. Reproduced in tests/test_critic_update.py. We use the standard TD(0) direction instead."),
 (4, "the loss function of the AN is", Y, "Eq. (17): actor minimises 1/2 Q^2. Since a_k = a_{k-1} + Delta a_k, both critic inputs depend on the actor: we use dQ/dDelta a + dQ/da."),
 (4, "values (ranging from 0.001 to 0.01)", B, "Table I: l_c = 0.01, l_a = 0.005 - used as given in the exact replication. With raw units the utility is ~1e-4, so at these rates neither network moves: MAACC stays equal to PID."),
 (5, "of the CN and the AN for the ith agent are UUB if the following", B, "UUB bounds the weight error but allows a large bound - consistent with the divergence we see for eq. (15)."),
 (7, "TABLE I", B, "ALL HYPERPARAMETERS: W=(1,2), Y=(0.5,0.05), R=(1e-4,1e-8), P=(2I,I), lambda=0.95, l_c=0.01, l_a=0.005, h_c=(12,14), h_a=(10,12). Used exactly in src/wwtp/maacc/exact.py. We missed this table at first - it is an image, so text extraction skipped it."),
 (7, "set to 45 s, and the total simulation duration is set to 14 days.", G, "Same protocol in our runs: 45 s, 14 days, first day excluded (PID warm-up)."),
 (7, "based on industry experience.", G, "Set points r = [1, 2] used as given."),
 (7, "diag{100, 100 000}", G, "Same PID gains. On the official BSM1 plant our PID gives IAE S_O,5 = 0.0082, S_NO,2 = 0.0095 (paper 0.0114, 0.0145): same order, ~30 % lower."),
 (8, "Trajectories of SO,5 and SNO,2 under the dry weather.", G, "Our version of this figure: docs/figures/readme_paper1_tracking.png - reimplemented on the official plant, the MAACC trajectory sits on top of the PID."),
 (8, "white noise with a mean of zero and a standard deviation of 3%", R, "NOT REPRODUCED. With this noise our PID gets 3x worse on the official plant (IAE 0.0082 -> 0.0245); the paper's PID only +10 % (0.0114 -> 0.0125). Their noise must reach the plant differently."),
 (9, "TABLE II", G, "NOT REPRODUCED AS PRINTED. Official BSM1 plant + official influent + Table I values, 20 runs: MAACC equals PID (0 % change, 0/20 runs better), for every critic-update reading. No initial-weight scale reproduces this table."),
 (9, "TABLE III", R, "Noise case: paper -86 % (O2), -69 % (NO3). Reimplemented as printed on the official plant: equal to PID."),
 (9, "and III are the average values obtained from 20 simulation", G, "We also ran 20 runs per case: 360 runs in total (results/summary_paper1_full_*.json)."),
 (9, "training is deemed successful", G, "Same success definition used. Reimplemented as printed: 0/20 in every case (MAACC = PID)."),
 (9, "the MAACC algorithm can reach over 90% across 20 simulation", Y, "Reimplemented as printed: 0/20 - the learned term never exceeds ~1e-4."),
]

P2 = [
 (1, "reward functions and by analysing the relationships among", Y, "Two rewards: A (effluent limits, eq. 19) and B (energy + ammonia, eq. 23). Code: src/wwtp/rl/rewards.py."),
 (7, "this paper selects S NH, S s, and S O as the input of the agent,", R, "PROBLEM FOUND: with an incremental action the actuator KLa5 is a hidden state - not Markov. On our rebuilt plant the published agent collapsed to zero air, so we add KLa5/240 to the state. On the official plant the published agent shut the air almost off mid-training but recovered (10.7 % saving, 1 seed)."),
 (7, "Figure 4 shows the variation of e" + FFL + "uent parameters when", G, "Reproduced: NH4 falls and Ntot rises with DO, COD/BOD/TSS barely move, energy rises (results/summary_fig4.json)."),
 (8, "environment at an interval of 45 s and obtain the required DO", R, "45 s steps with gamma = 0.99 give a ~75 min horizon, but effluent NH4 responds over hours. We hold each action 15 min."),
 (8, "the in\ufb02uent data are taken as the in\ufb02uent of WWTPs in the", G, "Same protocol: learn on week 1, evaluate on week 2."),
 (9, "the size of the replay bu" + "ﬀ" + "er is set to R = 30000;", B, "Same hyper-parameters: gamma 0.99, batch 256, tau 0.001, lr 1e-4 / 3e-4, actor/critic 256-256. Plus reward scaled by (1 - gamma) - the critic's dQ/da had the wrong sign for weeks without it."),
 (9, "ﬁnally we let β1 = 1, β2 = 0.42.", B, "beta2 = 0.42 used as given. AE entered as KLa5/240 (units not stated; raw AE ~3500 would swamp the 0.42 term)."),
 (10, "AE is reduced by 5.52% and", G, "Rain, official BSM1: our DDPG-B saves 7.1 % on average (3 seeds: 12.7 / 0.4 / 8.1 %), with more ammonium than the paper. PID AE 3691 vs paper 3666 (+0.7 %), EQ 8296 vs 8220."),
 (10, "AE is decreased by 7.24% and 7.22%", G, "Dry, official BSM1: PID AE 3728 vs paper 3698 (+0.8 %) - comparator matches. Our DDPG-B saves 10.6 % on average (seeds 10.5 / 5.8 / 15.7 %); seed 1 lands on the paper: AE 3513 vs 3430, DO 1.01 vs 1.03, S_NH 3.31 vs 3.26."),
 (10, "6.06% and 6.03%, respectively in", G, "Storm: no official storm file available to us. On our rebuilt plant the saving is 5.0 % (paper 6.06 %)."),
 (10, "appears to increase under DDPG-A and DDPG-B", R, "NOT REPORTED BY THE PAPER: how often the 4 g N/m3 ammonia limit is broken. Official plant: DDPG-B 56 % of the week (dry) vs 21 % for PID. The energy saving is bought with violations - this motivates our constrained, risk-aware agent."),
 (10, "Comparisons of AE and EQ under four di" + "ﬀ" + "erent control strategies", Y, "Official plant, mean S_O,5 dry: PID 1.998 (paper 2.000), DDPG-A 1.88 (1.415), DDPG-B 0.67 (1.032). Means over 3 seeds; DDPG is very seed-sensitive and the paper reports no seeds or spread."),
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

def write_notes(path):
    """docs/10_paper_study_notes.md from the same notes as the PDFs."""
    tag = {Y: "key claim", G: "matched", R: "problem / not reproduced", B: "parameter"}
    clean = lambda q: (q.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb04", "ffl")
                       .replace("\ufb00", "ff").rstrip(",.-"))
    L = ["# Study notes on the two base papers", "",
         "Page-by-page notes made while reimplementing each paper. Each row is a passage",
         "we highlighted in our annotated copy of the paper (the PDFs are copyrighted and not",
         "in this public repository; `tools/annotate_papers.py` rebuilds the annotated copies",
         "from your own PDFs, and `--notes` regenerates this file).", "",
         "Read the papers at their publishers:",
         "[paper 1, doi:10.1109/TII.2026.3659924](https://doi.org/10.1109/TII.2026.3659924) ·",
         "[paper 2, doi:10.1007/s11431-022-2403-8](https://doi.org/10.1007/s11431-022-2403-8).", "",
         "Tag: **matched** = we reproduced it; **problem** = it does not work as printed or",
         "did not reproduce; **key claim** = what the paper asserts; **parameter** = a setting we used.", ""]
    for title, items in (("Paper 1 — Wang et al., *Multiagent Adaptive Critic Control With Expert Knowledge for WWTPs*, IEEE T-II 2026", P1),
                         ("Paper 2 — Du et al., *DO concentration control in WWTP based on RL*, Sci China Tech Sci 2023", P2)):
        L += [f"## {title}", "", "| page | passage | tag | our note |", "|---|---|---|---|"]
        L += [f"| {p} | “{clean(q)}” | {tag[c]} | {n.replace('|', '/')} |" for p, q, c, n in items]
        L.append("")
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    if sys.argv[1] == "--notes":
        write_notes(sys.argv[2])
        sys.exit()
    run(sys.argv[1], sys.argv[3], P1,
        "SUMMARY - Exact replication: official BSM1 plant and influent, algorithm as printed, Table I values, 20 runs per case. PID matches the paper's PID row within ~30 % (dry). MAACC as printed EQUALS PID in all 180 runs: Table II/III NOT reproducible from the text. Legend: green = matched, red = problem / not reproduced, yellow = key claim, blue = parameter.")
    run(sys.argv[2], sys.argv[4], P2,
        "SUMMARY - Official BSM1 plant and influent: PID and fuzzy match the paper within 1 % (energy, effluent index). DDPG-B saves 10.6 % (dry) / 7.1 % (rain) on average vs paper 7.24 / 5.52 %, with more ammonium; seeds spread widely and one lands on the paper's numbers. Exactly as published (1 seed) it also ends near 10.7 %, after shutting the air almost off mid-training; our fixes (KLa5 in the state, 15-min decisions, reward scaled by (1 - gamma)) were needed on our rebuilt plant. Not reported by the paper: the ammonia limit is broken far more often than under PID. Legend: green = matched, red = problem / not reported, yellow = key claim, blue = parameter.")

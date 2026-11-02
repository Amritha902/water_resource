"""All architecture diagrams of the deck, every box placed by hand.  Writes d*.drawio."""

from dio import Diagram


def d0_plant():
    d = Diagram("BSM1 plant", 1200, 540)
    d.box(20, 8, 900, 34, "BSM1: the benchmark plant paper 1 controls (official IWA model)", "title")
    tanks = []
    for k, x in enumerate((90, 245, 400, 555, 710)):
        air = k >= 2
        lab = f"<b>Tank {k + 1}</b><br>{'aerobic' if air else 'anoxic'}<br>{'1333' if air else '1000'} m³"
        tanks.append(d.box(x, 190, 135, 150, lab, "tankair" if air else "tank"))
    setl = d.box(905, 190, 170, 110, "<b>Settler</b><br>10 layers", "settler")
    hop = d.box(945, 300, 90, 55, "", "hopper")
    d.line((20, 265), (88, 265), "")
    d.box(2, 275, 86, 44, "influent<br>≈ 18 446 m³/d", "text", "fontSize=11;align=center;")
    for a, b in zip(tanks, tanks[1:]):
        d.edge(a, b, ex=(1, 0.5), en=(0, 0.5))
    d.edge(tanks[4], setl, ex=(1, 0.5), en=(0, 0.682))
    eff = d.box(1120, 180, 70, 30, "effluent", "text", "fontSize=12;")
    d.edge(setl, eff, ex=(1, 0.14), en=(0, 0.5))
    # internal recycle Q_a (actuator 2)
    d.edge(tanks[4], tanks[0], "internal recycle  <b>Q_a</b>  (actuator 2: 0 – 92 230 m³/d)", kind="new",
           ex=(0.5, 0), en=(0.5, 0), pts=((777, 120), (157, 120)))
    # return sludge and waste
    d.line((990, 356), (157, 342), "return sludge Q_r", kind="fb", pts=((990, 440), (157, 440)))
    d.line((990, 405), (1150, 405), "waste sludge", kind="fb")
    # air
    for k, t in enumerate(tanks[2:], 2):
        x = 467 + 155 * (k - 2)
        lab = "air, KLa = 240" if k < 4 else "<b>KLa,5</b> (actuator 1: 0 – 240 1/d)"
        d.line((x, 395), (x, 342), "", kind="air" if k < 4 else "new")
        d.box(x - 80, 398, 160, 34, lab, "textc", "fontSize=11;" + ("fontColor=#0E3B2A;" if k == 4 else ""))
    # sensors
    d.box(258, 200, 110, 28, "S_NO,2 sensor", "white", "fontSize=12;fontStyle=1;")
    d.box(723, 200, 110, 28, "S_O,5 sensor", "white", "fontSize=12;fontStyle=1;")
    d.box(20, 470, 1160, 60, "<b>Loop 1</b>: KLa,5 → S_O,5 (dissolved oxygen in tank 5), set point 2 mg/L.   "
          "<b>Loop 2</b>: Q_a → S_NO,2 (nitrate in tank 2), set point 1 mg/L.<br>Influent: official dry, rain and "
          "storm files, 14 days at 15 min.  Control step 45 s.  Green: the two inputs the controllers move.", "legend")
    d.save("d0_plant.drawio")


def agent(d, x, y, title, u, z, loop):
    """One MAACC agent as printed in paper 1 (Section V, eqs. 7-19)."""
    g = d.box(x, y, 560, 230, title, "group")
    an = d.box(x + 20, y + 45, 150, 64, f"<b>Action network</b><br>π{loop}(z{loop}) → Δa{loop}", "paper")
    integ = d.box(x + 210, y + 45, 120, 64, f"integrate<br>a = a₋₁ + Δa{loop}", "paper")
    cn = d.box(x + 20, y + 140, 230, 70, f"<b>Critic network</b><br>Q{loop}(z, Δa{loop}, a{loop})<br>"
               "sees both loops", "paper")
    td = d.box(x + 285, y + 140, 255, 70, "one-step TD error<br>e = U + λ Q(k) − Q(k−1)<br>"
               "each step used once", "paper", "fontSize=13;")
    pid = d.box(x + 370, y + 45, 170, 64, f"<b>Expert PID</b> b{loop}<br>(incremental)", "paper")
    d.edge(an, integ, ex=(1, 0.5), en=(0, 0.5))
    d.edge(td, cn, "", kind="upd", ex=(0, 0.5), en=(1, 0.5))
    d.edge(cn, an, "dQ/da", kind="upd", ex=(0.3, 0), en=(0.3, 1))
    return g, an, integ, cn, td, pid


def d1_paper():
    d = Diagram("Paper 1 MAACC", 1200, 640)
    d.box(20, 8, 1100, 34, "Paper 1 (Wang et al., IEEE T-II 2026): multi-agent adaptive critic control (MAACC)",
          "title")
    a1 = agent(d, 20, 66, "Agent 1: dissolved oxygen", "KLa,5", "z1", 1)
    a2 = agent(d, 20, 340, "Agent 2: nitrate", "Q_a", "z2", 2)
    s1 = d.box(640, 170, 44, 44, "Σ", "sum")
    s2 = d.box(640, 450, 44, 44, "Σ", "sum")
    for (g, an, integ, cn, td, pid), s in ((a1, s1), (a2, s2)):
        gy = d_y = None
        top = {s1: 103, s2: 377}[s]
        sx = 662
        d.edge(integ, s, "a", ex=(0.5, 0), en=(0.5, 0), pts=((290, top), (sx, top)))
        py = {s1: 66 + 45 + 32, s2: 340 + 45 + 32}[s]
        sy = {s1: 192, s2: 472}[s]
        d.edge(pid, s, "b", ex=(1, 0.5), en=(0, 0.5), pts=((600, py), (600, sy)))
    plant = d.box(800, 230, 200, 200, "<b>BSM1 plant</b><br>(two coupled<br>subsystems)", "plant",
                  "fontSize=16;")
    d.edge(s1, plant, "u₁ = a₁ + b₁ = KLa,5", ex=(1, 0.5), en=(0, 0.25), pts=((740, 192), (740, 280)))
    d.edge(s2, plant, "u₂ = a₂ + b₂ = Q_a", ex=(1, 0.5), en=(0, 0.75), pts=((740, 472), (740, 380)))
    meas = d.box(1040, 290, 140, 80, "measured<br>S_O,5, S_NO,2<br>z = s − r", "white")
    d.edge(plant, meas, ex=(1, 0.5), en=(0, 0.5))
    d.edge(meas, a1[0], "z to both agents and PIDs", kind="fb", ex=(0.5, 0), en=(0.5, 0), pts=((1110, 48), (300, 48)))
    d.edge(meas, a2[0], "", kind="fb", ex=(0.5, 1), en=(0.5, 1), pts=((1110, 600), (300, 600)))
    d.box(765, 468, 325, 125, "<b>Claim</b> (Table II): oxygen error −95 / −95 / −92 %, nitrate −75 / −71 / −76 % "
          "against the PID (dry / rain / storm).<br><b>What we found</b>: rebuilt exactly as printed, it equals its "
          "PID in 80 of 80 runs (learning signal about 10⁻⁴).", "bad", "align=left;spacing=10;fontSize=13;")
    d.save("d1_paper.drawio")


def d2_ours():
    d = Diagram("Our controller", 1200, 560)
    d.box(20, 8, 1150, 34, "Our controller: the paper's PID, tuned, plus a daily learning feedforward (oxygen loop)",
          "title")
    r = d.box(10, 265, 90, 50, "set point<br>S_O,5 = 2", "textc", "fontSize=13;")
    s0 = d.box(120, 268, 44, 44, "Σ", "sum")
    d.box(100, 258, 20, 20, "+", "textc", "fontSize=15;")
    d.box(150, 316, 20, 20, "−", "textc", "fontSize=18;")
    pid = d.box(230, 245, 210, 90, "<b>Tuned PID</b><br>P and I tuned per loop<br>(robust: gains ×0.5)", "new")
    d.box(232, 227, 46, 18, "NEW", "tag")
    s1 = d.box(520, 268, 44, 44, "Σ", "sum")
    lim = d.box(620, 250, 160, 80, "<b>actuator limit</b><br>0 ≤ KLa,5 ≤ 240", "paper")
    plant = d.box(870, 225, 190, 130, "<b>BSM1 plant</b><br>tank 5", "plant", "fontSize=16;")
    ilc = d.box(360, 70, 300, 120, "<b>Daily ILC</b> (feedforward)<br>learns the PID's correction for each time "
                "of day and applies it 3 min earlier<br>bumpless at midnight, with anti&#8209;windup", "new", "fontSize=14;")
    d.box(362, 52, 46, 18, "NEW", "tag")
    sens = d.box(870, 410, 190, 80, "<b>sensor</b><br>official model:<br>delay + noise", "plant", "fontSize=14;")
    filt = d.box(620, 410, 160, 80, "<b>oxygen filter</b><br>τ = 0.75 min", "new")
    d.box(622, 392, 46, 18, "NEW", "tag")
    learn = d.box(290, 400, 300, 100, "<b>Optional learned correction</b><br>SAC-IL or repaired MAACC + safety "
                  "gate; adds nothing on the tuned PID + ILC", "learn", "fontSize=13;")
    d.edge(r, s0, ex=(1, 0.5), en=(0, 0.5))
    d.edge(s0, pid, "error", ex=(1, 0.5), en=(0, 0.5))
    d.edge(pid, s1, "u_PID", ex=(1, 0.5), en=(0, 0.5))
    d.edge(ilc, s1, "f(time of day)", kind="new", ex=(0.6067, 1), en=(0.5, 0))
    d.edge(pid, ilc, "", kind="upd", ex=(0.4286, 0), en=(0, 0.5), pts=((320, 130),))
    d.box(186, 142, 126, 24, "records u_PID", "text", "align=right;fontSize=13;")
    d.edge(lim, ilc, "", kind="upd", ex=(0.5, 0), en=(1, 0.5), pts=((700, 130),))
    d.box(712, 146, 220, 46, "at a limit: PID limits shift,<br>ILC skips that slot", "text", "fontSize=13;")
    d.edge(learn, s1, "", kind="learn", ex=(0.84, 0), en=(0.5, 1))
    d.edge(s1, lim, ex=(1, 0.5), en=(0, 0.5))
    d.edge(lim, plant, "KLa,5", ex=(1, 0.5), en=(0, 0.5))
    d.edge(plant, sens, "S_O,5", kind="fb", ex=(0.5, 1), en=(0.5, 0))
    d.edge(sens, filt, "", kind="fb", ex=(0, 0.5), en=(1, 0.5))
    d.edge(filt, s0, "measured S_O,5", kind="fb", ex=(0.5, 1), en=(0.5, 1), pts=((700, 535), (142, 535)))
    d.save("d2_ours.drawio")


def d3_ilc():
    d = Diagram("Daily ILC", 1200, 440)
    d.box(20, 8, 1150, 34, "How the daily ILC learns (one update per night, no neural network)", "title")
    k = d.box(30, 70, 320, 160, "<b>Day k</b><br>the PID controls; the ILC records the PID's input u_PID for every "
              "time of day (1 920 slots of 45 s)", "paper")
    m = d.box(440, 70, 320, 160, "<b>Midnight update</b><br>f ← f + β · shift(u_PID − mean)<br>shifted 3 min "
              "earlier, smoothed ±22 min; β = 1 (tuned on dry week 1)", "new")
    k1 = d.box(850, 70, 320, 160, "<b>Day k+1</b><br>f(time of day) is added to the PID's output, ahead of the "
               "daily load peak, so the PID corrects less", "paper")
    d.edge(k, m, ex=(1, 0.5), en=(0, 0.5))
    d.edge(m, k1, ex=(1, 0.5), en=(0, 0.5))
    d.edge(k1, k, "repeated every day", kind="upd", ex=(0.5, 1), en=(0.5, 1), pts=((1010, 275), (190, 275)))
    d.box(30, 315, 550, 115, "<b>Bumpless transfer</b> (round 2)<br>at midnight the same change is taken out of "
          "the PID's state, so the input does not jump. Largest oxygen deviation near midnight: 0.053 → 0.011 mg/L",
          "new", "align=left;spacing=14;")
    d.box(620, 315, 550, 115, "<b>Anti-windup</b> (after held-out test 1)<br>the PID's limits move with f, and "
          "slots at an actuator limit teach nothing. Under overload (new data): error −39 %; normal weather unchanged",
          "new", "align=left;spacing=14;")
    d.save("d3_ilc.drawio")


def d4_repaired():
    d = Diagram("Repaired MAACC", 1200, 470)
    d.box(20, 8, 1150, 34, "Repaired MAACC: the paper's structure kept, its training fixed (a fair learning baseline)",
          "title")
    d.box(20, 60, 520, 390, "Kept from paper 1", "group")
    for i, (t, s) in enumerate([("one agent per loop", "oxygen, nitrate"),
                                ("action network outputs an increment", "u = a + b, with the PID prior b"),
                                ("critic sees both loops and both actions", "the coupling between loops"),
                                ("starts exactly as the PID", "last layer initialised at zero")]):
        d.box(40, 100 + 85 * i, 480, 68, f"<b>{t}</b><br>{s}", "paper")
    d.box(600, 60, 580, 390, "Repaired (standard fixes, as in MATD3)", "groupnew")
    for i, (t, s) in enumerate([("replay buffer", "every step kept and re-used (paper: used once)"),
                                ("target networks + twin critics", "smaller of two critics, smoothed targets"),
                                ("normalised inputs and reward", "paper's learning signal was about 10⁻⁴"),
                                ("bounded increment, warm-up day", "tanh limit; one day of exploration first")]):
        d.box(620, 100 + 85 * i, 540, 68, f"<b>{t}</b><br>{s}", "new")
    d.save("d4_repaired.drawio")


def d5_method():
    d = Diagram("Evaluation", 1200, 520)
    d.box(20, 8, 1150, 34, "How we tested: train on one week, freeze, then tests nothing was tuned on", "title")
    steps = [("Official BSM1 data", "dry, rain, storm<br>14 days each", "paper"),
             ("Tune and train", "dry weather,<br>days 0–7 only", "new"),
             ("Freeze", "settings and<br>trained networks", "new"),
             ("Test week", "days 7–14, every<br>weather (315 runs)", "new")]
    ids = []
    for i, (t, s, k) in enumerate(steps):
        ids.append(d.box(30 + 290 * i, 70, 240, 100, f"<b>{t}</b><br>{s}", k))
    for a, b in zip(ids, ids[1:]):
        d.edge(a, b, ex=(1, 0.5), en=(0, 0.5))
    steps2 = [("Protocol 1 pushed to git", "before any run", "new"),
              ("Held-out test 1", "new influent (BSM2),<br>official sensors, delay", "new"),
              ("Two fixes", "ILC anti-windup;<br>noise-robust PID", "new"),
              ("Protocol 2 pushed", "before any run", "new")]
    ids2 = []
    for i, (t, s, k) in enumerate(steps2):
        ids2.append(d.box(30 + 290 * i, 230, 240, 100, f"<b>{t}</b><br>{s}", k))
    for a, b in zip(ids2, ids2[1:]):
        d.edge(a, b, ex=(1, 0.5), en=(0, 0.5))
    d.edge(ids[3], ids2[0], "", ex=(0.5, 1), en=(0.5, 0), pts=((1020, 200), (150, 200)))
    h2 = d.box(30, 390, 530, 100, "<b>Held-out test 2</b><br>new BSM2 windows (raw and clarified), new sensor-noise "
               "seeds; 432 runs", "new")
    au = d.box(640, 390, 530, 100, "<b>Audit</b><br>re-runs match to machine precision; metrics recomputed "
               "independently; 76 of 76 checks pass", "new")
    d.edge(ids2[3], h2, "", ex=(0.5, 1), en=(0.5, 0), pts=((1020, 360), (295, 360)))
    d.edge(h2, au, ex=(1, 0.5), en=(0, 0.5))
    d.save("d5_method.drawio")


if __name__ == "__main__":
    d0_plant()
    d1_paper()
    d2_ours()
    d3_ilc()
    d4_repaired()
    d5_method()

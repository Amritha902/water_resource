"""Rebuild workshop/SPEECHIFY_OUTLINE.md from the deck: one entry per slide, in order.

Keeps every narration already written (matched by slide title) and writes one for dividers,
code maps, quizzes and any slide without one.

    python workshop/tools/build_outline.py
"""

import html as H
import re
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
OUT = WS / "SPEECHIFY_OUTLINE.md"
LESSON = {"lesson1": "the four training steps", "lesson2": "download, read, check and fix",
          "lesson3": "choose, split, scale and cut", "lesson4": "shapes, baselines, linear, MLP and CNN",
          "lesson5": "train, validate, test once and save", "lesson6": "environment, reward, act and learn"}


#: narration written by hand for slides whose text alone does not make a good voice-over
EXTRA = {
    "The plant and its two control loops": (
        "This is the plant as the project drew it. Water flows from left to right. Tanks 1 and 2 get no air, so "
        "bacteria there remove nitrate. Tanks 3 to 5 are aerated, so other bacteria turn ammonium into nitrate, and the "
        "internal recycle carries that nitrate back to tank 1. The settler lets the sludge sink, and most of it goes back "
        "to the start. Our controllers move two things: the air in tank 5, to hold 2 milligrams of oxygen per litre, and "
        "the recycle pump, to hold 1 milligram of nitrate per litre in tank 2."),
    "First the basics, then our project": (
        "The session has three parts. Part 1 is the basics: how any machine-learning project goes, one step per slide, on a "
        "simple example with no plant in it. Part 2 does the same steps on our research project, with its real files and "
        "code. Part 3 is reinforcement learning and the project's controller. Each part ends with a few questions and a "
        "notebook you run yourself."),
    "Every machine-learning project follows the same steps": (
        "Every machine-learning project follows the same twelve steps. Collect the data, clean it, decide what goes in and "
        "what comes out, split it into training, validation and test, scale it, and feed it in. Then choose a model, train "
        "it, check its size against the data, choose its settings, test it once, and save it. Our example for part 1: a "
        "town wants to predict tomorrow's water demand from the weather."),
    "Where data comes from, and the first look": (
        "Data comes from files, databases, public datasets or simulations. Whatever the source, write down where it came "
        "from, which version, the units of every column, and whether you may share it. Reading it is one line of pandas. "
        "Then look before doing anything else: the shape, the type of each column, the first rows, and the minimum and "
        "maximum of every column. In our example that first look already shows a demand ten times too big and a negative "
        "rainfall."),
    "Cleaning: find what is wrong, fix only what you can explain": (
        "Cleaning means finding what is wrong and fixing only what you can explain. Wrong types are converted, duplicate "
        "rows dropped, short gaps filled from their neighbours, long gaps left out, impossible values set to unknown, and "
        "spikes found by comparing each value with the median of its neighbours. Keep a list of every fix, so someone else "
        "can rebuild your dataset."),
    "Decide what goes in (X) and what comes out (y)": (
        "One row is one example. The inputs, X, are what we know today: tomorrow's weather forecast, whether tomorrow is a "
        "weekend, and today's demand. The target, y, is tomorrow's demand. Only put in X what you will really have when you "
        "make the prediction. And know the shape: here 1094 examples of 4 inputs and 1 target."),
    "Train, validation and test: three separate piles": (
        "We split the examples into three piles. The model learns from the training pile. We make our choices with the "
        "validation pile. The test pile gives the final score, and we look at it once. When the data comes in time order, "
        "split by time: a random split puts near-copies of training examples into the test, and the score looks far better "
        "than it is."),
    "Put every input on a similar scale": (
        "Inputs on very different scales confuse a network: a flow of twenty thousand would drown a temperature of 25. So "
        "we take the log of quantities that grow by factors, then standardise: subtract the mean, divide by the standard "
        "deviation. The mean and standard deviation come from the training data only, and the same numbers are used for "
        "everything after, so the scaler is saved with the model."),
    "How the data gets into the model: tensors and batches": (
        "The data reaches the model as tensors, arrays PyTorch can compute with and trace back for gradients. A data loader "
        "shuffles the training examples and serves them in batches of 32. The model turns each batch of 32 by 4 inputs into "
        "32 guesses at once. One pass through all the batches is an epoch."),
    "What is a model?": (
        "A model is a function with knobs. Here it is a straight line, y hat equals w times x plus b. The input x is what we "
        "know, y hat is the model's guess, and the two knobs w and b are its parameters. The red lines are its mistakes; "
        "their average square is the loss. Training turns each knob a little in the direction that lowers the loss, again "
        "and again. A neural network is the same thing with thousands of knobs."),
    "Three kinds of learning: it depends on what answers you have": (
        "There are three kinds of learning. Supervised: you have the right answer for every example, a number for "
        "regression or a class for classification. Unsupervised: there is no answer, and the model finds structure, such "
        "as groups of similar samples. Reinforcement: no right answers at all, only a reward after acting. Our demand "
        "example is supervised regression."),
    "The shape of your data picks the kind of model": (
        "The shape of the data picks the kind of model. A table: start with a linear model, trees, or a small network. A "
        "sequence: a convolution over time, a recurrent network or a Transformer. An image: a convolutional network. The "
        "output picks the last layer and the loss. And always start simple: a baseline with no learning, then a linear "
        "model, and only then something bigger."),
    "What training code looks like": (
        "This is what training code looks like. For each batch: run the model, compute the loss, clear the old gradients, "
        "compute new ones with backward, and let the optimiser turn the knobs. After every epoch, measure the loss on the "
        "validation data with learning switched off, and keep the knobs from the best epoch. The curves show why: training "
        "loss keeps falling, but validation loss turns up when the model starts to memorise."),
    "Counting parameters, and comparing with your data": (
        "How big should the model be? Count its knobs. A layer from n inputs to m outputs has n times m weights plus m "
        "biases. Our small model has 97 knobs for 765 training examples, so it cannot memorise much. A model with a million "
        "knobs for the same data could learn every example by heart. The shapes fix the first and last layers; the amount "
        "of data limits what goes in between."),
    "Parameters are learned; settings are chosen by you": (
        "Parameters are learned by training. Settings, also called hyperparameters, are chosen by you: the number of units, "
        "the learning rate, the batch size, the number of epochs. Try a few and keep the one with the lowest validation "
        "loss. Here the middle size wins: 4 units are too few, 64 start to memorise. Bigger is not automatically better."),
    "What testing code looks like": (
        "Testing comes last, after the settings are chosen. Put back the best knobs, switch learning off, predict the test "
        "examples, and turn the predictions back into real units. Then compare with a baseline, the simplest guess there "
        "is: tomorrow equals today. Our model is off by 44 thousand litres a day, the baseline by 57. Test once: if you change "
        "the model after looking, the test has become validation."),
    "One notebook, twelve steps: the skeleton of every ML notebook": (
        "Here is the whole of part 1 on one slide: the skeleton of every machine-learning notebook. Imports, collect, clean, "
        "inputs and target, split, scale, feed, model, train, size, settings, test, save. Every notebook you will meet, "
        "including our project's, is these blocks in this order. When a notebook confuses you, ask which of the twelve "
        "steps a cell belongs to."),
    "Notebook 0: the skeleton": (
        "Your turn. Open notebook 0 and run it from top to bottom; it takes about ten seconds. Then change one thing at a "
        "time: remove today's demand from the inputs, make the model much bigger, use a huge learning rate, or split at "
        "random, and see what each change does to the validation and test errors."),
    "Our data: the official BSM1 influent files": (
        "Now our own data. The official BSM1 influent files: what flows into the plant, every 15 minutes for 14 days, "
        "which is 1345 rows of 15 columns: the time, 13 concentrations and the flow. There are three files, for dry, rain "
        "and storm weather. The first week is the same dry week in all three, which matters when we split."),
    "Cleaning our files, and a fix only physics can find": (
        "Cleaning our files. The usual problems from part 1 come back: text rows with titles and units, missing values, "
        "duplicate times and spikes. But the official files had one problem no statistical check finds. The inert COD, "
        "S_I, stays at 30 even when rain doubles the flow. Rain water is clean, so it must dilute the sewage. The project "
        "applies the dilution rule, S_I equals 30 times the dry flow divided by the flow, and writes the fix down."),
    "Turn one long series into many (input, target) examples": (
        "Our data is one long time series, not a table, so we cut it into examples. A window slides along the series: the "
        "last 24 hours are the input, the next 2 hours are the target. The input always ends where the target begins, so "
        "the model only sees the past. Stacked up, the inputs have the shape N by 96 time steps by 3 channels, and the "
        "targets N by 8. We split by weather and time: dry and rain to train, the first storm to validate, the second "
        "storm to test."),
    "Choose, split, scale, cut: cell by cell": (
        "Notebook 1, part C, running cell by cell: choosing the inputs, splitting by time, scaling with the training "
        "numbers, cutting the windows and checking their shapes."),
    "Notebook 1: collect, clean, prepare": (
        "Your turn: notebook 1, parts A, B and C. Download and open the official files, check and clean them, apply the "
        "dilution fix, then choose the inputs, split, scale and cut the windows. If the notebook gives trouble, the lesson "
        "2 and lesson 3 scripts do the same in the terminal."),
    "Our shapes, and the models in the project": (
        "Which model for our data? The table from part 1 answers it. The input is a sequence, 96 time steps of 3 "
        "measurements, and the output is 8 numbers. A sequence in and numbers out: a one-dimensional convolution over time, "
        "with a linear last layer. But first the baselines, then a linear model, then an MLP. On the right are the other "
        "models in the project: different shapes, so different models."),
    "Notebook 2: choose and train": (
        "Your turn: notebook 2. Build the baselines, a linear model, a small network and a causal CNN, train them with a "
        "validation set, overfit on purpose, then test once and save the model with its scaler. Fill in the table of "
        "parameters and test errors as you go."),
    "When there is no correct answer to copy": (
        "Part 3 starts with the idea of reinforcement learning. When we control the air, there is no right answer to copy. "
        "So an agent tries actions, the plant responds, and a reward scores what happened. The reward is where you write "
        "down what you want. Paper 2's reward pays for air and for ammonium above the limit; its agent saved energy and "
        "broke the ammonium limit far more often. It followed the reward exactly as written."),
    "The parts of an actor-critic agent (DDPG, SAC)": (
        "The parts of an actor-critic agent: the environment, an actor network that picks the action, a critic network "
        "that estimates the future reward, a replay buffer of past experience, slow target copies of the networks, and "
        "noise for exploring. The critic is trained exactly like the regression in part 1, against a target it makes for "
        "itself."),
    "Notebook 3: reinforcement learning in brief": (
        "Your turn: notebook 3. A one-tank aeration environment driven by the real daily load, a PID as the expert, a "
        "network that copies it, and a Q-learning agent that learns from the reward alone. Then change the energy weight "
        "in the reward and watch what the agent does with the oxygen."),
    "The second held-out test: two fixes, one open problem": (
        "These are the project's latest results. The first held-out test found two flaws, so two fixes were made: "
        "anti-windup for the daily learning control, and a robust PID with a filter on the oxygen reading and half the "
        "gains. They were tested on new influent and new sensor noise, with the plan committed before any run. "
        "Anti-windup removed the learning control's harm under overload and cost nothing in normal weather. The robust PID "
        "never failed and was 83 percent better than the paper's PID on new influent. But with the official sensor noise, "
        "nothing we built beats the paper's own gentle PID. That is the open problem, and it is reported."),
}


def txt(s):
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def main():
    old = OUT.read_text(encoding="utf8")
    intro = old[:old.index("## Deck") + len("## Deck")]
    narr = {}
    for blk in re.split(r"(?=### Slide \d+ ·)", old)[1:]:
        t = re.match(r"### Slide \d+ · (.*)", blk).group(1).strip()
        m = re.search(r"\*\*Narration:\*\* (.*)", blk, re.S)
        if m:
            narr[t] = m.group(1).strip().split("\n\n")[0]
    h = (WS / "slides" / "index.html").read_text(encoding="utf8")
    deck = (WS / "slides" / "deck.js").read_text(encoding="utf8")
    qblock = deck[deck.index("const QUIZ = {"):deck.index("const QSTATE")]
    quiz = {}
    for key, body in re.findall(r"(\w+): \[(\[.*?\])\],?(?=\s*\n\s*(?:\w+: \[\[|\}))", qblock, re.S):
        quiz[key] = re.findall(r'\["(.*?)", "(.*?)"\]', body)
    secs = re.findall(r'<section class="(slide[^"]*)"([^>]*)>(.*?)</section>', h, re.S)
    intro = re.sub(r"\d+ slides", f"{len(secs)} slides", intro)
    out = [intro, "\n\n"]
    for i, (cls, attrs, body) in enumerate(secs, 1):
        m = re.search(r"<h[12][^>]*>(.*?)</h[12]>", body, re.S)
        title = txt(m.group(1)) if m else f"Slide {i}"
        k = re.search(r'<div class="kicker">(.*?)</div>', body)
        kick = txt(k.group(1)) if k else ""
        lis = [txt(x) for x in re.findall(r"<li[^>]*>(.*?)</li>", body, re.S)][:6]
        anim = (re.search(r'data-anim="(\w+)"', attrs) or [None, ""])[1]
        key = (re.search(r'data-key="(\w+)"', attrs) or [None, ""])[1]
        bul = lis
        if anim == "quiz":
            qs = quiz.get(key, [])
            bul = [f"Q: {q} → A: {a}" for q, a in qs]
            lead = "Questions learners ask, with short answers." if key == "faq" else \
                f"{kick.split('·')[0].strip()} ends with four questions. Ask the room first, then reveal each answer."
            nar = lead + " " + " ".join(f"{q} {a}" for q, a in qs)
        elif anim == "codemap":
            scr = re.search(r"examples/(\w+\.py)", body).group(1)
            bul = [f"The same steps ({LESSON[key]}), each in its own colour, in four panels: the hands-on script, "
                   "the notebook, and the project's own code", f"Run it yourself: python workshop/examples/{scr}"]
            nar = (f"Where is the code? These four panels are real files. Each step, {LESSON[key]}, "
                   "has one colour, and it lights up in every panel at once: first in the short script you can run, "
                   "then in the notebook, then in the project's own code. What you just learned is exactly what the "
                   "research code does.")
        elif "divider" in cls:
            goal = txt(re.search(r"In this part</h3><p[^>]*>(.*?)</p>", body, re.S).group(1))
            hands = txt(re.search(r"Hands-on</h3>(.*?)</div>", body, re.S).group(1))
            bul = [f"In this part: {goal}", f"Hands-on: {hands}"]
            nar = f"{kick.split('·')[0].strip()}: {title}. {goal}"
        else:
            nar = EXTRA.get(title) or narr.get(title) or (f"{title}. " + (" ".join(lis[:3]) if lis else txt(body)[:300]))
        if not bul:
            p = re.search(r'<p class="sub"[^>]*>(.*?)</p>', body, re.S)
            bul = [txt(p.group(1))] if p else []
        out.append(f"### Slide {i} · {title}\n" + (f"- Section: {kick}\n" if kick else "")
                   + "".join(f"- {b}\n" for b in bul) + f"- Image: `stills/slide{i:02d}.jpg`\n"
                   + ("- Interactive in the HTML deck: use the live deck for this slide.\n" if anim else "")
                   + f"\n**Narration:** {nar}\n\n")
    OUT.write_text("".join(out), encoding="utf8")
    print("wrote", OUT, len(secs), "slides")


if __name__ == "__main__":
    main()

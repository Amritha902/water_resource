# From plant data to a trained model

This is the whole workshop written out as a lesson, in the order of the slides. Read it once from top to bottom and you can teach the session yourself. For each slide you get what is on it, what to say, the plant physics where it matters, and the code, a few lines at a time, with what each line does and why it is there.

Every code block below is copied from the repository when this guide is built, and every output block comes from running the lesson scripts at that moment. If the code changes, rebuild the guide (`python workshop/tools/build_guide.py`) and it follows.

How to use it on the day. Keep the deck (`workshop/slides/index.html`) on the projector and this guide on your laptop. The right arrow moves on. On the interactive slides it first steps through the slide: the next notebook cell, the next block of code, the next quiz answer. Each lesson ends with the same terminal command the scholars can run, so if a notebook breaks on someone's machine, the script still works.

The timing is 90 minutes. Lessons 2 and 5 are the long ones because they hold the main hands-on sessions.

## Before the lessons

### Slide 1 · From plant data to a trained model

On the slide is the plant, animated, with a slider for the air blown into the last tank.

Say: "Today we learn machine learning the way it is used in research, on one real project from start to finish. The project is about a wastewater treatment plant. Bacteria clean the water, and they need oxygen. We supply that oxygen by blowing air. More air gives cleaner water and costs more electricity. Deciding how much air to blow, minute by minute, is the control problem our models try to solve."

Move the slider while you talk. The oxygen reading changes, and so does the energy figure underneath. That one trade-off is the whole project in miniature.

### Slide 2 · One lesson, seven steps, 90 minutes

Read the seven steps out loud once. Then point out the rhythm, because it repeats in every lesson: the idea with an animation, where the code is, the code running, three questions, and the scholars' turn. Tell them now that every step has a one-line terminal command, so nobody gets left behind if their notebook misbehaves.

### Slide 3 · Every word you will hear today

Thirty-two words, each with a plain meaning, a comparison from daily life and where it appears in the project. Do not read them all. Click four: *model*, *loss*, *overfitting* and *reinforcement learning*. Then tell the room this slide is their dictionary for the day and they can come back to it any time.

### Slide 4 · What the project is about

Go through the five bullets in order. The table on the right matters more than it looks: the project contains four different kinds of machine learning, and the scholars will see all four today.

- Forecasting the inflow two hours ahead is **regression**: the answer is a number.
- Warning before ammonium breaks the legal limit is **classification**: the answer is yes or no.
- Predicting how the plant will respond is a **sequence model**: the answer depends on everything that happened before.
- Deciding how much air to blow is **reinforcement learning**: there is no right answer to copy, only a score afterwards.

### Slide 5 · The plant and its two control loops

This is the project's own drawing of the plant, so spend two minutes on the physics. The scholars do not need chemistry, but they need to know why the two control loops exist.

#### The physics

Wastewater carries two kinds of pollution the plant must remove: organic matter (measured as COD, chemical oxygen demand) and nitrogen, mostly as ammonium.

- **Tanks 3, 4 and 5 are aerated.** Air is blown in, so there is dissolved oxygen. Here one group of bacteria (the nitrifiers) turns ammonium into nitrate. This needs oxygen: roughly 4.6 g of oxygen for every gram of ammonium nitrogen.
- **Tanks 1 and 2 are not aerated** (anoxic). With no oxygen, a second group of bacteria breathes the oxygen locked in nitrate instead. They turn nitrate into nitrogen gas, which bubbles off harmlessly. They need food to do it, and the incoming COD is that food.
- **The internal recycle Q_a** pumps water from tank 5 back to tank 1. That is how the nitrate made in the aerated tanks gets back to the anoxic tanks to be removed.
- **The settler** lets the bacteria (the sludge) sink. Clean water leaves over the top. Most of the sludge goes back to tank 1 so the bacteria are not lost, and a little is wasted every day so they do not pile up.

Some numbers to quote. The tanks hold 1000 + 1000 + 1333 × 3 = 5999 m³, and the average inflow is 18 446 m³ per day. So the water spends 5999 / 18 446 = 0.33 days in the tanks, about 8 hours.

**Loop 1, oxygen.** The controller sets KLa₅, the oxygen transfer coefficient of tank 5. KLa is how fast oxygen from the air dissolves into the water, per day. Its range is 0 to 240 per day. The goal is 2 mg/L of oxygen in tank 5. Why 2? The nitrifiers slow down when oxygen is scarce. The model uses a factor S_O / (K_OA + S_O) with K_OA = 0.4 mg/L. At 2 mg/L that factor is 2 / 2.4 = 0.83, so they run at 83 % of full speed. Going higher buys little and costs a lot of air.

**Loop 2, nitrate.** The controller sets the recycle flow Q_a, 0 to 92 230 m³/d, to hold 1 mg/L of nitrate in tank 2. Too little recycle and nitrate escapes in the effluent. Too much and the pump wastes energy, and it carries oxygen back into the tanks that are meant to be oxygen-free.

The whole plant is a standard computer model called BSM1, Benchmark Simulation Model no. 1. Everyone in this field tests their controllers on the same simulated plant, so results can be compared. Our project uses the official version (the `bsm2-python` package), wrapped in `src/wwtp/official/plant.py`.

### Slide 6 · How the project was built: one script per step

Press the right arrow and walk through the boxes in order. Each box is one script in `experiments/`, and the panel shows the command that runs it and the file it writes. The story to tell:

1. Rebuild the official plant and its data.
2. Rebuild three published controllers exactly as their papers print them, and check their claims. Two of the three did not work as printed.
3. Build our own controller and two strong baselines to compare it with.
4. Test everything on data nothing was tuned on, twice, with the test plan committed to git before each test was run.
5. Audit every number. 76 of 76 checks pass.

The point to land: research code is a chain of small scripts, each writing a result file that the next one reads, so every number in the report can be traced back to a command.

## Lesson 1. What is a model?

### Slide 7 · Lesson 1 of 7

Read the goal: "A model is a function with knobs. Training turns the knobs, in four steps that every model uses." Everything else today builds on this.

### Slide 8 · What is a model?

Press **train** and let the line fit itself while you talk.

Say: "A model is a function with knobs. Here the function is a straight line, ŷ = w · x + b. The input x is the inflow. The output ŷ is our guess of the ammonium load. The two knobs are w, the slope, and b, the offset. Those two numbers are the model's parameters. Training changes them and nothing else.

How wrong is the line? Each red line is one mistake. Square each mistake, take the average, and you have one number called the loss. Training means turning each knob a little in the direction that makes the loss smaller, and doing that again and again. A neural network is exactly this with thousands of knobs and a bendier function."

### Slide 9 · Four steps train every model

This slide shows the same four steps in four places at once: the lesson script, notebook 2, the project's flow forecaster and the project's reinforcement-learning critic. Press the right arrow and each step lights up in all four panels in the same colour. The point: the 2-knob toy and the 171 211-knob forecaster are trained by the same four lines.

#### The code: `lesson1_knobs.py`

Run it with `python workshop/examples/lesson1_knobs.py`. First the data.

{{code:workshop/examples/lesson1_knobs.py|rng = np.random.default_rng(0)|y = 0.7 * x}}

- `default_rng(0)` makes a random number generator with seed 0, so everyone in the room gets the same numbers.
- `x` is 20 inflow values between 0 and 1 (already scaled).
- `y` is the true load. We make it ourselves with a hidden rule, slope 0.7 and offset 0.15, plus a little noise. The model is never told 0.7 or 0.15. It has to find them. That is why we invent the data here: we know the right answer, so we can check the model got it.

Now training by hand, in plain numpy.

{{code:workshop/examples/lesson1_knobs.py|w, b = 0.0, 0.0|the hidden rule was}}

- `w, b = 0.0, 0.0`: the knobs start at zero. The model knows nothing yet.
- `lr = 0.5` is the learning rate, the size of each turn of the knobs.
- `for step in range(201)` repeats the four steps 201 times.
- **Step 1, guess.** `pred = w * x + b` runs the model on all 20 inputs at once.
- **Step 2, how wrong.** `loss = np.mean((pred - y) ** 2)` is the mean squared error, the average of the squared mistakes.
- **Step 3, which way.** The gradient says how the loss changes when you nudge each knob. For this loss, calculus gives `mean(2·(pred − y)·x)` for w and `mean(2·(pred − y))` for b. If the gradient for w is positive, turning w up makes things worse, so we turn it down.
- **Step 4, turn the knobs.** `w = w - lr * grad_w` steps against the gradient, downhill.

Part B does exactly the same with PyTorch, the way the project's code does it.

{{code:workshop/examples/lesson1_knobs.py|xt = torch.tensor|opt.step()}}

- `torch.nn.Linear(1, 1)` is a layer with one input and one output: one weight (our w) and one bias (our b).
- `torch.optim.SGD(model.parameters(), lr=0.5)` hands the knobs to an optimiser. SGD is plain gradient descent, the same rule as Part A.
- `opt.zero_grad()` clears the gradients from the last step. PyTorch adds gradients up by default, so forgetting this line is a classic bug.
- `loss.backward()` computes every gradient automatically. This replaces the two hand-written lines of Part A, and it works for a network of any size.
- `opt.step()` turns every knob.

The output, from the run that built this guide:

{{run:lesson1_knobs}}

Read it with the room. The loss falls from 0.31 to 0.0011, and w and b end near 0.67 and 0.17. They do not land exactly on 0.7 and 0.15 because of the noise we added: with 20 noisy points, this is the best line. PyTorch arrives at the same answer.

> Try this with the room: change `lr = 0.5` to `1.2` and run again. The loss grows instead of shrinking, because each step jumps over the bottom of the valley and lands higher on the other side. This prepares question 3 of the quiz.

### Slide 10 · Check yourself

Ask each question to the room before you reveal the answer.

1. Which parts does training change? Only w and b. The data and the shape of the function stay fixed.
2. What does a large loss mean? The guesses are far from the truth: the knobs are not good yet, or a straight line cannot fit this data.
3. What happens with a learning rate that is too big? The steps overshoot and the loss explodes.

## Lesson 2. Collect and clean the data

### Slide 11 · Lesson 2 of 7

Goal: download data so anyone can get the same file again, read it without falling into traps, check it, and fix what is wrong, including the physics. This lesson takes 20 minutes with its hands-on session.

### Slide 12 · Where the data comes from

The curves draw themselves as if a sensor were logging them, and rows appear in the table on the right.

Say: "Our data is what flows into the plant: the flow rate and 13 concentrations, every 15 minutes, for 14 days. That is 1345 rows. There are three files, one per weather: dry, rain and storm. Look at the first week. It is the same in all three files. Keep that in mind, because if we split carelessly we will test on data we trained on."

The rain file has one long rain period around days 8.4 to 10.4. The storm file has two short storms, around day 8.8 and day 11, with flows up to 60 000 m³/d, more than three times the dry average.

### Slide 13 · Load it once, record where it came from, cache it

This is the project's loading code. Four habits, one per bullet.

- **Pin the version.** The download address contains a commit hash. A file behind a hash never changes, so anyone can rebuild the same dataset years later.
- **Keep only the data rows.** Spreadsheets have titles and units in them. The project keeps a row only if its first cell is a number.
- **Cache it.** Reading Excel is slow. The project saves the numbers as `.npy`, which loads in milliseconds.
- **Respect the licence.** These files are under the GPL, so the repository downloads them when needed instead of storing copies.

### Slide 14 · Cleaning: look before you learn

The table on the left is invented to show the usual problems. Each right-arrow press highlights one problem and its fix.

1. Text rows: drop them, but write the units down somewhere.
2. Missing values: short gaps can be filled by drawing a straight line between the neighbours (interpolation). Long gaps should be left empty, because filling them would invent data.
3. Duplicate timestamps: keep one, then check the time step is a steady 15 minutes.
4. Spikes: a value ten times too big is usually a unit or typing error. Compare each value with the median of its neighbours, and flag it if it is far away.
5. The physics check. This is the one the project actually needed. More below.
6. Missing columns the plant model needs, such as total suspended solids.

#### The physics of the S_I fix

S_I is inert soluble COD: organic matter the bacteria cannot eat, so it passes straight through. In the spreadsheets it is 30 g/m³ in every row of every file, even when rain doubles the flow. Rain water is clean, so it must dilute the sewage. If the flow doubles because of rain, the concentration of anything already in the sewage should halve. The official rain file shipped with the `bsm2-python` package follows exactly this rule, S_I = 30 · Q_dry / Q. The project applies it to the rain and storm files (`plant.py`, line 89) and writes it down in `docs/DATASET.md`.

No statistical check finds this. There are no missing values, no duplicates and nothing negative. Only someone who knows what the numbers mean can see it is impossible.

### Slide 15 · Put everything on one time grid

The sensors give a value every 15 minutes, but the controller acts every 45 seconds. Between two samples we keep using the last one, a zero-order hold. That is 20 control steps per sample. The animation shows the held values filling in step by step.

### Slide 16 · Download, read, check, fix

The same four steps in the lesson script, notebook 1 and the project. Press through them.

#### The code: `lesson2_load_and_clean.py`

Run it with `python workshop/examples/lesson2_load_and_clean.py`.

{{code:workshop/examples/lesson2_load_and_clean.py|# ── 1 DOWNLOAD|urllib.request.urlretrieve(URL + dry_path.name, dry_path)}}

- `DATA / "Inf_rain_2006.xlsx"` builds a path to the data folder. `Path` objects join folders with `/`, which also works on Windows.
- `if not path.exists():` downloads only the first time. After that the file on disk is used.
- `urlretrieve` fetches the file from the pinned address in `URL`.

{{code:workshop/examples/lesson2_load_and_clean.py|# ── 2 READ|after reading properly}}

- `pd.read_excel(path, header=None)` reads the sheet without saying which row holds the column names. The first row of this file holds names like `t`, `Si`, `Ss`. Because one row is text, pandas treats every column as text (type `object`). Nothing crashes, which is what makes this trap dangerous.
- `header=0` says row 0 is the header. Now the 1345 rows below it are data.
- `d.columns = COLS` gives the columns names we choose and understand.
- `d.astype(float)` makes every column a number, and fails loudly if something is not a number. Loud failures are good.

{{code:workshop/examples/lesson2_load_and_clean.py|# ── 3 CHECK|S_I values in the whole file}}

- `np.diff(d.t) * 1440` is the gap between consecutive times, converted from days to minutes (1440 minutes per day). We expect exactly one value, 15.
- `d.isna().sum().sum()` counts missing values in the whole table.
- `d.t.duplicated().sum()` counts repeated timestamps.
- `(d.drop(columns="t") < 0).sum().sum()` counts negative values. Concentrations and flows cannot be negative.
- The last line prints every distinct value of S_I. There is only one, 30, which is the physics problem.

{{code:workshop/examples/lesson2_load_and_clean.py|# ── 4 FIX (a)|round(float((x.S_NH}}

- `x = d.copy()` works on a copy, so we can compare the repair with the original.
- We damage the copy on purpose: three readings go missing and one becomes ten times too big.
- `rolling(9, center=True).median()` is the median of each reading and its 8 neighbours, 4 on each side. One spike cannot move a median, so the median shows what the value should be.
- `spike = (x.S_NH - med).abs() > 0.5 * med` flags a reading more than 50 % away from its neighbourhood.
- `interpolate(limit_area="inside")` fills the holes with straight lines between the values on both sides.

One caution for the room: `center=True` uses readings from after the moment being checked. That is fine for cleaning a stored file. A controller running live cannot see the future and must use only past values.

{{code:workshop/examples/lesson2_load_and_clean.py|# ── 4 FIX (b)|S_I after the dilution fix}}

`30.0 * dry.Q / d.Q` is the dilution rule: when the flow is twice the dry-weather flow, S_I is half of 30.

{{run:lesson2_load_and_clean}}

The repair finds the one spike, and the largest remaining error is 1.5 g/m³ against values around 30. After the fix S_I goes down to 10 g/m³ at the peak of the rain, when the flow is three times the dry flow.

### Slide 17 · How the project loads and corrects the official data

A code tour of `load_influent` in `src/wwtp/official/plant.py`. Press through five blocks:

1. A second data source, the BSM2 influent, used only for the held-out tests. It now has a version run through a simple primary clarifier, added for held-out test 2.
2. The raw table, from the cache.
3. The 22-column layout the plant simulator expects.
4. The S_I fix, one line.
5. The derived and constant columns: suspended solids, the flow, and the temperature of 15 °C.

### Slide 18 · Collect and clean, cell by cell

Notebook 1 running inside the slide. Each press runs one cell and shows its real output. The panel on the right explains the idea behind the cell. Use this slide if the room is slow to get started: they can watch the first cells run before trying themselves.

### Slide 19 · Check yourself

1. Why did every column come out as text? One row of names makes pandas read the whole column as text.
2. All the checks passed. Can the data still be wrong? Yes. S_I stays at 30 while rain doubles the flow. Only physics catches it.
3. When may you fill a gap by interpolation? Only when it is short, here up to one hour.

### Slide 20 · Your turn: notebook 1, parts A and B (12 minutes)

Scholars open `workshop/notebooks/01_data_collect_clean_prepare.ipynb`. Walk around the room. The usual problems: they skip a cell, or their kernel was started in the wrong folder. The menu item *Kernel, Restart Kernel and Run All Cells* fixes most of it. Anyone stuck can run the lesson 2 script instead.

## Lesson 3. Prepare the data

### Slide 21 · Lesson 3 of 7

Goal: choose inputs and a target, split by time, scale with numbers from the training data only, and cut the series into examples.

### Slide 22 · Transform and scale the inputs

Click the three buttons: raw flow, log of flow, standardised.

Say: "Flow is about 20 000; ammonium is about 30. Inside a model, every input is multiplied by a weight and the results are added. The big number would drown out the small one. So we bring every input to the same scale, around zero with a spread of one.

First the log. Flows and loads change by factors: a storm multiplies the flow by three. Taking the log turns multiplication into addition, so the storm no longer stretches the whole scale. Then we standardise: subtract the mean, divide by the standard deviation.

Now the most important rule of the lesson. The mean and the standard deviation must come from the training data only. We then reuse exactly those two numbers for the validation data, the test data and the real plant. If you compute them from all the data, the test set has leaked into training."

### Slide 23 · Turn one long series into many examples

The animation slides a window along the flow curve. The blue part is the input, the last 24 hours. The orange part is the target, the next 2 hours. Each position of the window is one example. Stack them and you get an array of shape (examples, 96 time steps, 3 channels).

The input always ends exactly where the target begins. The model only ever sees the past, as it will on the real plant.

### Slide 24 · Split by time, not at random

Click *random split*. Red outlines appear on test windows that overlap a training window.

Say: "Two neighbouring windows share 95 of their 96 samples. Split them at random and almost every test window has a near-copy in the training set. The score looks wonderful and means nothing. So we split by time: learn on the past, check on later data, test on data nothing was tuned on."

How the project does it: paper 2 learns on week 1 and tests on week 2. The warning model uses separate simulated plants for training, validation and test. Paper 1 got two held-out tests with their plans committed before running them.

### Slide 25 · Choose, split, scale, cut

The four steps in the lesson script, notebook 1 and the project's own dataset code. Press through them.

#### The code: `lesson3_prepare.py`

Run it with `python workshop/examples/lesson3_prepare.py`. It writes the file that lessons 4 and 5 read.

{{code:workshop/examples/lesson3_prepare.py|# ── 1 CHOOSE|LOOKBACK, HORIZON = 96, 8}}

- `CHANNELS` lists the inputs: flow, ammonium and readily biodegradable COD. These are things a real plant can measure at the inlet. We do not give the model anything a plant could not know.
- `LOOKBACK = 96`: 96 samples of 15 minutes is 24 hours of history. One whole day, so the model can see the daily pattern.
- `HORIZON = 8`: 8 samples is 2 hours ahead.

{{code:workshop/examples/lesson3_prepare.py|# ── 2 SPLIT|"test": [("storm"}}

The split is by the time of the target, in days. Training uses the dry file from day 1 (day 0 only serves as history) and the rain file's second week. Validation is storm 1, test is storm 2. The model never sees a storm while it trains. That is on purpose: the test asks whether it can cope with something new.

{{code:workshop/examples/lesson3_prepare.py|# ── 3 SCALE|return (np.log(x) - MEAN) / STD}}

- `train_rows` gathers every raw sample the training windows touch, including their 24 hours of history (`lo - 1`).
- `MEAN, STD` are computed from the log of those rows only.
- `leaky` is what the mean would be if the test data were included. We print it only to show the difference. The numbers move by about 0.02, which looks small, but it is information from the future that should not be there.
- `scale(x)` applies the same transform anywhere: log, minus the training mean, divided by the training standard deviation.

{{code:workshop/examples/lesson3_prepare.py|# ── 4 WINDOWS|return np.array(X), np.array(Y), np.array(T)}}

- For each start position `s`, `first = s + LOOKBACK` is the row where the target begins.
- The `if` skips windows whose target lies outside this split, so no window crosses from training into test.
- `z[s:first]` is the input: 96 rows, 3 channels. `z[first:first + HORIZON, 0]` is the target: the next 8 values of channel 0, the flow.

{{run:lesson3_prepare}}

Read the shapes out loud: 1906 training examples, each 96 steps by 3 channels, each with 8 numbers to predict. The script also measures the leak a random split would cause: 98 % of test windows would sit right next to a training window.

### Slide 26 · Prepare, cell by cell

Notebook 1, part C, running cell by cell.

### Slide 27 · Check yourself

1. Why not split at random? Neighbouring windows share 95 of 96 samples, so the test would contain near-copies of training examples.
2. Where do the mean and standard deviation come from? The training data only.
3. Read the shape (1906, 96, 3). 1906 examples, 96 time steps each, 3 measurements per step.

### Slide 28 · Your turn: notebook 1, part C (10 minutes)

Ask them to think about the question on the slide while it runs: training has no storm, the test is a storm. What will happen?

## Lesson 4. Choose the model

### Slide 29 · Lesson 4 of 7

Goal: let the shape of the input and output choose the kind of model, look inside a network layer by layer, and count its knobs.

### Slide 30 · The input and output decide the model

Say: "Start from the shapes, not from the fashionable model. If your input is a table, start with a linear model or gradient-boosted trees. If it is a sequence, use a model that knows about time order. If your output is a number, the last layer is linear and the loss is squared or absolute error. If it is a yes-or-no probability, use a sigmoid and cross-entropy."

Then click through the project's six models on the right. Each card gives the shapes, the model, the loss and the reason. The last one is the published MAACC controller, which has only 10 to 14 knobs that learn. Remember that for lesson 7.

### Slide 31 · What happens inside a neural network, layer by layer

Press *next step* by hand, slowly. This slide is where most people finally understand what a network is.

Say: "Three inputs come in: flow, ammonium and COD. Each neuron in the first layer multiplies every input by its own weight, adds them, adds its bias, and gets one number z. Then it applies ReLU: if z is negative it becomes zero, otherwise it stays. The next layer does the same with these results. The last neuron gives the forecast. Blue lines are positive weights, red are negative, and thicker lines are bigger weights. Training only changes those numbers."

A whole layer at once is h = ReLU(W·x + b), one matrix multiplication. That is why networks run fast on hardware built for matrices.

### Slide 32 · Counting parameters

Type layer sizes or click the presets. A dense layer from n inputs to m outputs has n × m weights and m biases.

Our small network, 288 → 64 → 64 → 8, has (288·64 + 64) + (64·64 + 64) + (64·8 + 8) = 23 176 knobs for 1906 training examples. That is about 12 knobs per example, common for a network, but it means we must watch the validation curve. Click *too big* to show 1.3 million knobs: far more than examples, so it can memorise.

### Slide 33 · A causal convolution

A small filter of 3 weights slides along time. Each output looks at 3 inputs, and only from the past: padding is added on the left so nothing from the future gets in. With dilations 1, 2, 4, each layer looks twice as far back, so a few layers cover a whole day with very few weights. The project's forecaster sees 253 steps back, more than its 96-step input.

### Slide 34 · A recurrent network carries a memory

The same cell runs at every time step. It reads the new input and the memory from the previous step, and writes a new memory. The project's digital twin uses one. The memory stands in for things no sensor measures, such as how many bacteria are in the tanks.

### Slide 35 · Shapes, baselines, linear, MLP, CNN

The five steps of the lesson in four panels. Press through them.

#### The code: `lesson4_models.py`

Run it with `python workshop/examples/lesson4_models.py`. It reads the file lesson 3 wrote.

{{code:workshop/examples/lesson4_models.py|d = np.load(|count = lambda m}}

- `np.load` opens the prepared file. `X` and `Y` hold the training and validation arrays.
- `flow(z)` undoes the scaling: multiply by the standard deviation, add the mean, then `exp` undoes the log. We always report errors in m³/d, the units an engineer understands.
- `mae(p)` is the mean absolute error for each of the 8 horizons.
- `count(m)` adds up the size of every parameter tensor: the number of knobs.

{{code:workshop/examples/lesson4_models.py|# ── 2 BASELINES|print(f"2 BASELINE}}

Two rules with no learning. *Persistence* says the next 2 hours equal the last value: `X["val"][:, -1:, 0]` is the last flow of each window, repeated 8 times. *Same time yesterday* uses the fact that the input is exactly one day long, so the value 24 hours before each target is `X["val"][:, :8, 0]`.

{{code:workshop/examples/lesson4_models.py|# ── 3 LINEAR|print(f"3 LINEAR}}

`reshape(len(...), -1)` flattens each 96 × 3 window into one row of 288 numbers, a table. `Ridge` is linear regression with a small penalty on large weights, which keeps it stable. It has 288 × 8 weights and 8 biases.

{{code:workshop/examples/lesson4_models.py|# ── 4 MLP|str(layer)[:26]}}

- `nn.Sequential` stacks layers in order: flatten, a linear layer to 64, ReLU, another to 64, ReLU, and a final linear layer to 8 outputs.
- The loop passes one example through the layers one at a time and prints the shape after each. It also counts how many values ReLU set to zero. With random starting weights about half are switched off.

{{code:workshop/examples/lesson4_models.py|# ── 5 CNN|never before it (causal)}}

- `CausalBlock`: `F.pad(x, (self.pad, 0))` adds `pad` zeros at the start of the time axis and none at the end. That is what makes it causal. `x + ...` adds the input back (a residual connection), which helps deep stacks train.
- The network is a 1 × 1 convolution from 3 to 24 channels, then six blocks with dilations 1 to 32.
- The test at the end changes one moment, time step 50, and checks which outputs change. Only outputs from step 50 onwards change. That proves no output looked into the future.

{{run:lesson4_models}}

The numbers to point at: persistence is good at 15 minutes and poor at 2 hours. Yesterday is the same at every horizon. The linear model beats both at every horizon. The CNN has fewer knobs than the MLP (10 808 against 23 176) for the same input, because one small filter is reused at every time step.

### Slide 36 · Shapes, baselines and models, cell by cell

Notebook 2, sections 1 to 6, running. The layer-by-layer cell is worth showing even if you ran the script.

### Slide 37 · Check yourself

1. How many knobs has a layer from 64 to 8 units? 64 × 8 + 8 = 520.
2. What does ReLU do? Keeps positive numbers, sets negative ones to zero. That bend lets a network learn curved patterns.
3. Why is the convolution causal? Padding goes on the left only, so each output sees only the present and the past.

## Lesson 5. Train and evaluate

### Slide 38 · Lesson 5 of 7

Goal: train with a validation set, recognise overfitting, beat the baselines, test once, and save the model with its scaler.

### Slide 39 · The loss says what "wrong" means

Toggle the curves. Squared error (MSE) punishes big mistakes hard and aims at the mean. Absolute error (MAE) is calmer about outliers and aims at the median. The pinball loss is lopsided: at τ = 0.9, guessing too low costs 9 times more than guessing too high, so the best answer is the value exceeded only 10 % of the time. Train three of these and you get a band, which is what the project's forecaster outputs. Cross-entropy is the loss for classes and yes-or-no questions.

### Slide 40 · Gradient descent: walk downhill on the loss

Try the three learning rates. At 0.02 the ball barely moves. At 0.15 it walks to the bottom. At 0.5 it bounces from side to side and flies off. This is the explosion from lesson 1, drawn. In practice the project uses AdamW, which adapts the step size for each knob, plus a learning-rate schedule and gradient clipping.

### Slide 41 · The training loop

The circle animates the five stages: a batch, forward, loss, backward, update. Below it the two loss curves are drawn epoch by epoch. Training loss keeps falling. Validation loss falls, then starts to rise. That turning point is where the model starts to memorise, and that epoch is the one we keep.

### Slide 42 · Train, validate, test once, save

The four steps in the lesson script, notebook 2, the project's early-warning training code and the project's forecaster loader.

#### The code: `lesson5_train_and_evaluate.py`

Run it with `python workshop/examples/lesson5_train_and_evaluate.py`. It takes about 5 seconds.

{{code:workshop/examples/lesson5_train_and_evaluate.py|def mlp(hidden):|return nn.Sequential(*layers, nn.Linear(n, 8))}}

`torch.manual_seed(0)` fixes the random starting weights. Without it every run starts from different knobs and ends with slightly different numbers, and scholars ask why theirs do not match the slide. With it, everyone gets the same result.

{{code:workshop/examples/lesson5_train_and_evaluate.py|def fit(model, epochs|return np.array(curve), best_epoch}}

- `DataLoader(..., batch_size=128, shuffle=True)` serves the training examples in shuffled batches of 128. One pass through all of them is an epoch.
- `AdamW` is the optimiser. `weight_decay` gently pulls every weight towards zero, a mild brake on memorising.
- Inside the batch loop are lesson 1's four steps in two lines: guess and loss in one, then `zero_grad`, `backward`, `step`.
- `model.eval()` and `torch.no_grad()` switch learning off while we measure. We measure both the training and the validation loss after every epoch.
- `if va < best:` keeps a copy of the weights from the epoch with the lowest validation loss. `copy.deepcopy` matters: without it we would keep a reference to weights that go on changing.
- At the end `load_state_dict(best_state)` puts the best epoch back.

{{code:workshop/examples/lesson5_train_and_evaluate.py|big = mlp((512, 512))|(it memorised)}}

The overfitting experiment: a network with 415 000 knobs trained on only 200 examples, with no weight decay.

{{code:workshop/examples/lesson5_train_and_evaluate.py|# ── 3 TEST|print("3 TEST (once)}}

The test set, used once, at the end, for all three models. Nothing about the models is changed after this line.

{{code:workshop/examples/lesson5_train_and_evaluate.py|# ── 4 SAVE|no storm in the training data}}

- `torch.save` writes the weights together with the mean and standard deviation and the layer sizes. A model saved without its scaler cannot be used on new raw measurements.
- The reload builds the same network from the saved sizes and loads the weights.
- `z = (np.log(raw) - blob["mean"]) / blob["std"]` scales raw readings with the saved numbers, never with numbers computed from new data.

{{run:lesson5_train_and_evaluate}}

What to say about these numbers:

- The small MLP's best epoch is late, and its validation loss stays well above its training loss. The validation data is a storm the model never saw.
- The big network shows overfitting clearly. Training loss falls from about 0.35 to under 0.01. Validation loss is best early and is worse at the end. It learned the 200 examples by heart.
- On the test storm, the linear model and the MLP are close, both far better than persistence. With about 1900 examples, a model with 2312 knobs is as good as one with 23 176.
- The reloaded model forecasts the calm period almost exactly and overshoots the storm. No storm was in the training data. More varied data would help more than a bigger model.

### Slide 43 · Training, overfitting and the test, cell by cell

Notebook 2, sections 5 to 9, running. The overfitting cell draws both curves side by side.

### Slide 44 · A model is only good compared with something simple

The left side lists the baselines. The bar chart on the right shows the project's own lesson. Read the bars from the README: a published learning controller, rebuilt exactly, did nothing beyond its PID, and our own learned controller was matched by a carefully tuned PID with the daily ILC. That is a result, not a failure, and it belongs in the report.

### Slide 45 · Check yourself

1. Training loss falls and validation loss rises. Overfitting: the model memorises. Keep the best epoch, regularise, use a smaller model or more data.
2. Why use the test set only once? Any choice made after looking at it turns it into training data.
3. What must be saved with the weights? The scaler and the settings.

### Slide 46 · Your turn: notebook 2 (20 minutes)

The longest hands-on session. Ask everyone to fill in the table on the slide: parameters and test error at 2 hours for each model. When most have finished, ask who had the best model, and point out that the linear model is usually at or near the top.

## Lesson 6. Reinforcement learning

### Slide 47 · Lesson 6 of 7

Goal: learn from a reward instead of right answers, then see the project's agents.

### Slide 48 · When there is no correct answer to copy

The loop animates: the agent sees the state, picks an action, the tank responds, a reward comes back.

Say: "For control there is no right answer to copy. Nobody can tell us the perfect amount of air for every minute. So the agent tries actions, the plant responds, and a reward scores what happened. The reward is where you write down what you want. Paper 2's reward is on the slide: pay for the air you use, and pay for ammonium above the limit of 4. That agent saved 9 % energy, and it broke the ammonium limit far more often than the PID. It did exactly what the reward allowed."

### Slide 49 · The parts of an actor-critic agent

The table lists the parts: the environment, an actor network that picks actions, a critic network that estimates the future reward of an action, a replay buffer of past experience, slow target copies of the networks, and exploration noise. The code on the right is one DDPG update. Point at the critic's loss: it is a squared error against a target, the same kind of regression as lesson 5.

### Slide 50 · Environment, reward, act, learn

The four steps in the lesson script, notebook 3, the project's environment and reward, and the project's RL training loop.

#### The physics of the toy tank

Lesson 6 uses one aerated tank, small enough to train in seconds. Its oxygen follows a mass balance:

change of oxygen per day = KLa × (8 − S) − OUR

- KLa × (8 − S) is oxygen dissolving from the air. 8 g/m³ is the saturation level at 15 °C, the most the water can hold. The further the oxygen S is below saturation, the faster it dissolves.
- OUR is the oxygen uptake rate, what the bacteria consume. It follows the incoming load, which we take from the real dry-weather file: flow × (ammonium + a fifth of the COD), scaled so its average is 700 g/m³ per day.

If nothing changed, S would settle where the two terms balance: S_eq = 8 − OUR / KLa. With the average load, KLa = 117 per day holds 2 mg/L, because 8 − 700/117 = 2.0. The tank approaches S_eq with a time constant of 1/KLa. At KLa = 120 per day that is 1/120 of a day, 12 minutes. Each step of the toy is 5 minutes, so the tank answers quickly.

#### The code: `lesson6_rl.py`

Run it with `python workshop/examples/lesson6_rl.py`. It takes about 15 seconds.

{{code:workshop/examples/lesson6_rl.py|d = pd.read_excel(|LOAD = np.repeat}}

`d[:, 14]` is the flow, `d[:, 10]` ammonium and `d[:, 2]` the readily biodegradable COD. Dividing by the mean makes the load a factor around 1. `np.repeat(..., 3)` turns 15-minute samples into 5-minute steps by holding each value three times.

{{code:workshop/examples/lesson6_rl.py|class OxygenTank:|return self.state(), reward, self.k >= 288}}

- `ACTIONS` are the five moves allowed: lower the air by 20 or 5, hold, raise by 5 or 20.
- `reset()` starts a day: step 0, air at 120, oxygen at 2 mg/L.
- `state()` is what the agent sees: the oxygen error (S − 2), the previous error, the air setting as a fraction of 240, and the current load.
- `step()` applies the move, clipped to 0 to 240. `our` is the bacteria's demand with 5 % random noise. Then the oxygen is moved towards S_eq with the exact solution of the mass balance over 5 minutes: `S_eq + (S − S_eq) · e^(−KLa · Δt)`, where Δt = 5/1440 of a day. Using the exact solution keeps the simulation stable at any step size.
- The reward is minus the oxygen error, minus the cost of air. One day is 288 steps of 5 minutes.

{{code:workshop/examples/lesson6_rl.py|pid = lambda s:|print(f"1 ENVIRONMENT}}

The expert, an incremental PID. The error is e = 2 − S, which is `-s[0]`, and the previous error is `-s[1]`. The change of air is 80 × (e − e_previous) + 25 × e. The first term reacts to how fast the error changes, the second pushes steadily until the error is gone. No data, no training.

{{code:workshop/examples/lesson6_rl.py|E, K, L = np.array(|greedy = lambda Q}}

Q-learning keeps a table. To use a table the state must be a cell, so `cell(s)` sorts the error into 7 bins, the air setting into 16 and the load into 6. `greedy(Q)` is the policy that picks the action with the highest value in the current cell.

{{code:workshop/examples/lesson6_rl.py|def q_learning(|return best[1]}}

- `Q = np.zeros((7, 16, 6, 5))`: one value for every cell and every action, all starting at zero.
- `eps` is the exploration rate. It starts at 1 (every move random) and falls to 0.05 over the first 60 % of training.
- Each episode is one random day from week 1.
- **Act.** With probability `eps` the agent tries a random move, otherwise the best one it knows.
- **Learn.** `target = r + 0.9 × best value of the next cell` is the Bellman target: the reward now plus the discounted value of where we landed. The table entry moves 20 % of the way towards it.
- Every 100 episodes the greedy policy is tested on day 6 and the best table is kept, the same idea as keeping the best epoch.

{{code:workshop/examples/lesson6_rl.py|ff = {"last": None}|print(f"    BASELINE}}

The fair baseline. The agent sees the load, so we give the PID the same information the engineer's way. From the mass balance, holding 2 mg/L needs KLa = OUR / (8 − 2) = 700 × load / 6. The feedforward adds the change of that value to the PID's move.

{{run:lesson6_rl}}

What to say:

- Q-learning beats the plain PID on a day it never trained on (0.235 against 0.304 mg/L average error).
- The PID with one line of physics does as well (0.230), with no training and no bad days of exploration.
- With air made 50 times more expensive in the reward, the agent switches the air off and the oxygen collapses. Nothing in the reward said the oxygen must stay up.

### Slide 51 · Reward, DDPG update and SAC-IL update, in the project's code

A code tour in three tabs. The paper 2 reward, two lines. One DDPG update: replay batch, Bellman target, critic regression, actor step, soft target update. One SAC-IL update: a quarter of each batch comes from the expert's demonstrations, and the imitation term never fades below a floor.

{{code:src/wwtp/rl/ddpg.py|# eq. (9): y = r + gamma|self.opt_actor.step()}}

- `y = r + γ · (1 − d) · Q_target(s′, μ_target(s′))` is the Bellman target, made with the slow target networks so it does not jump every step. `(1 − d)` drops the future at the end of an episode.
- `loss_c` is the squared error between the critic's estimate and that target. This is lesson 5's regression.
- `loss_a` is minus the critic's value of the actor's action. Minimising it means the actor climbs the critic. The small penalty on `pre` keeps the `tanh` output from getting stuck at a limit.

### Slide 52 · Reinforcement learning, cell by cell

Notebook 3 running, including the learning curve and the comparison on the unseen rain day.

### Slide 53 · Start from the expert, then improve, behind a safety gate

The project's controller in four boxes. The expert PID is run on dry week 1 and its moves recorded. A network copies it (behaviour cloning, which is supervised learning). Soft actor-critic improves on that copy using the reward. A safety gate switches the learned part off whenever the inputs leave the range seen in training. The code on the left is the copying step, a plain regression.

### Slide 54 · Check yourself

1. What is the difference between a reward and a label? A label is the right answer. A reward only scores what happened.
2. Why does the agent sometimes act at random? To explore. Without trying new moves it can never find a better one.
3. What did the agent do when air was made very expensive? It switched the air off.

### Slide 55 · Your turn: notebook 3 (12 minutes)

The question on the slide is the best discussion of the day: change the energy weight in the reward and watch the oxygen. Ask where the behaviour flips. In our runs the agent still holds the oxygen at a weight of 3, starts cutting the air at 4, and gives up on the oxygen at 5.

## Lesson 7. Our project's code

### Slide 56 · Lesson 7 of 7

Goal: read the published methods and our controller in the project's own code, and see what the latest tests found.

### Slide 57 · Papers 1 and 3 exactly as printed

Two code tours. Paper 1's MAACC is a small adaptive-critic controller added on top of a PID.

{{code:src/wwtp/maacc/exact.py|def step(self, z: np.ndarray) -> float:|return a}}

- The action network `alpha @ bipolar(eta * zi)` gives a change of input, which is added up into `a`. Because `alpha` starts at zero, the controller starts exactly as its PID.
- The critic estimates the value `q` and the utility `U` scores the moment, with the weights printed in the paper's Table I.
- The update of `alpha` is the learning rate times `q` times the slope of the critic. With the printed weights that product is about 10⁻⁴, so `alpha` hardly moves. In 80 of 80 runs, MAACC equals its PID to within 1 %. The code is correct. The method, as printed, cannot learn on this plant.

Paper 3's IOQL repeats 50 critic updates and 60 actor updates on each single sample. That many steps on one point makes it unstable: it either stays at its PID or the oxygen collapses.

### Slide 58 · ILC, the control step, the safety gate and the training script

Our controller in four files. The first is the daily ILC, iterative learning control. Each night it looks at what the PID had to do during the day, and the next day it does part of that in advance.

{{code:src/wwtp/sacil/ilc.py|def _learn(self):|self.f = new}}

- `_learn` runs once a day, at midnight.
- `ok` marks the moments when the blower was free. When the blower is flat out or off, what the PID asked for is not what the plant got, so those moments teach nothing. If more than half the day was spent at a limit, the ILC keeps yesterday's profile (`continue`). This is the anti-windup, added after held-out test 1.
- `dev` is the PID's deviation from its daily mean: the part of its work that repeats every day.
- `step` takes a share `beta` of that and moves it a few steps earlier (`np.roll`), so the air arrives just ahead of the load peak instead of just after it.
- The moving average smooths the profile over the day, treating midnight as joined to the next morning.

The control step adds the other half of the anti-windup.

{{code:src/wwtp/sacil/sacil.py|aw = self.ilc is not None and self.ilc.antiwindup|sat = (self.u <= lo}}

The PID's own limits are shifted by the feedforward `ff`, so PID plus feedforward together can reach either end of the blower's range and no more. Without this the PID keeps adding correction while the blower is already at its limit (it winds up), and then needs a long time to unwind.

The third tab is the safety gate.

{{code:experiments/33_paper1_study.py|def act(o, e):|return net(np.asarray(o, np.float32))}}

If any input is more than 10 % outside the range seen in training, or the PID needs a bigger move than the learned part may make, the learned part is switched off for this step and the PID acts alone.

### Slide 59 · The second held-out test: two fixes, one open problem

This slide holds the project's latest results (`docs/PAPER1_HELDOUT2.md`). The plan for this test was committed to git before any of its 432 runs.

Held-out test 1 had found two flaws. Under overload the ILC made things worse, and every high-gain controller failed with the official sensor noise. Two fixes were made:

1. **ILC anti-windup**, the code above.
2. **A robust PID**: a 0.75-minute filter on the oxygen reading and half the gains, chosen on dry week 1 with spare noise seeds that no test uses.

They were then tested on new BSM2 influent windows, with and without a primary clarifier, and with new sensor-noise seeds. Read the table on the slide:

- Anti-windup works on new data. Under overload it removes the ILC's harm, 39 % better, and costs nothing in normal operation.
- The robust PID never failed and is 83 % better than the paper's PID on new influent, with about the same blower movement as the paper's PID (522 against 501).
- With the official sensor noise, 0.25 g/m³ on a set point of 2 mg/L, nothing built in the project beats the paper's own gentle PID. That is the open problem. Next ideas are a Kalman filter on the oxygen reading, or switching to lower gains when the noise is high.

Say: "This is what honest research looks like. Two fixes worked and are reported. One problem is still open and is reported too. A test that only uses clean data, as paper 1 did, would have hidden all of this."

### Slide 60 · Check yourself

1. Why does MAACC as printed equal its PID? Its learning signal is about 10⁻⁴, so its networks hardly move.
2. What does the safety gate do? Switches the learned part off on inputs outside the training range.
3. What did held-out test 2 show? Anti-windup fixed overload, the robust PID never failed, and under sensor noise nothing beat the paper's PID.

### Slide 61 · Questions learners ask

Eight questions with short answers, collected by going through the lesson as a beginner. Open the ones the room cares about.

### Slide 62 · What this project teaches about doing ML research

End on the eight habits. Look at the data first. Let the shapes choose the model. Split honestly. Beat simple baselines. Reproduce published work before claiming to improve it. Use several seeds. Write the test down before you run it. Save the model with everything it needs. Then take questions.

## After the session

Everything is in the repository and in the zip bundle. The scholars can rerun any lesson with one command from `workshop/examples/`. The notebooks are also web pages with every output. The project's own documents (`docs/PAPER1_STUDY.md`, `docs/PAPER1_HELDOUT2.md`, `docs/PAPER1_AUDIT.md`) take them as deep as they want to go.

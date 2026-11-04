# From plant data to a trained model

This is the whole workshop written out as a lesson, in the order of the slides. Read it once from top to bottom and you can teach the session yourself. For each slide you get what is on it, what to say, the plant physics where it matters, and the code, a few lines at a time, with what each line does and why it is there.

Every code block below is copied from the repository when this guide is built, and every output block comes from running the lesson scripts at that moment. If the code changes, rebuild the guide (`python workshop/tools/build_guide.py`) and it follows.

How to use it on the day. Keep the deck (`workshop/slides/index.html`) on the projector and this guide on your laptop. The right arrow moves on. On the interactive slides it first steps through the slide: the next notebook cell, the next block of code, the next quiz answer.

The session has three parts and takes 90 minutes. Part 1 (40 minutes) is the basics: every step of a machine-learning project, one per slide, on a simple example with no plant in it. Part 2 (30 minutes) does the same steps on our project's real data and code. Part 3 (20 minutes) is reinforcement learning and the project's controller. Keep the project out of part 1. If a scholar asks about the plant there, say it is coming in part 2.

## Opening

### Slide 1 · From plant data to a trained model

On the slide is the plant, animated, with a slider for the air blown into the last tank.

Say: "Today we learn machine learning the way it is used in research, on one real project from start to finish. The project is about a wastewater treatment plant. Bacteria clean the water, and they need oxygen. We supply that oxygen by blowing air. More air gives cleaner water and costs more electricity. Deciding how much air to blow, minute by minute, is the control problem our models try to solve."

Move the slider while you talk. The oxygen reading changes, and so does the energy figure underneath. That one trade-off is the whole project in miniature.

### Slide 2 · First the basics, then our project

Read the three parts out loud. Say: "First we learn the recipe on an easy problem, so the ideas are not buried under plant chemistry. Then we cook the same recipe with our research data, where every step is harder and more interesting." Point at the notebook links: the scholars can read every notebook as a web page with all its results if their laptop gives trouble.

### Slide 3 · Every word you will hear today, explained simply

Thirty-four words, each with a plain meaning, a comparison from daily life and where it appears in the project. Do not read them all. Click four: *model*, *loss*, *overfitting* and *reinforcement learning*. Then tell the room this slide is their dictionary for the day and they can come back to it any time.

## Part 1. The basics

### Slide 4 · The basics

The example for the whole of part 1: a town wants to know tomorrow's water demand, and it has three years of daily temperature, rainfall and demand. Everything in part 1 is in `workshop/notebooks/00_ml_skeleton.ipynb`, which makes its own data, so it needs no download and runs in about ten seconds.

### Slide 5 · Every machine-learning project follows the same steps

Press the right arrow twelve times; each step appears in turn. Read them as a story: "We get the data, clean it, decide what goes in and what comes out, split it into piles, scale it, feed it in, choose a model, train it, test it, check its size, tune its settings and save it." The colours group them: grey is getting the data ready, blue is the model and its training, green is judging and keeping it. Tell the room that the next eighteen slides take these steps one at a time, and that most research time goes into the grey ones.

### Slide 6 · Where data comes from, and the first look

Data comes from files (CSV, Excel, logs), databases, public datasets and simulations. Whatever the source, write four things down: where the file came from, which version or date, the units of every column, and whether you may share it.

Then the code. `pd.read_csv` reads the file into a table (a DataFrame); `read_excel` and `read_sql` do the same for other sources. The four lines after it are the first look, and they are not optional:

- `df.shape` gives rows and columns. Is it as big as you expected?
- `df.dtypes` gives the type of each column. A column of numbers that comes out as text (`object`) means something in it is not a number.
- `df.head()` shows the first rows. Titles, units and notes often hide there.
- `df.describe()` gives the minimum, maximum, mean and spread of each column. Read the minimum and maximum against common sense.

Press the arrow for the last line: in our example `describe()` already shows a demand ten times too big and a negative rainfall.

### Slide 7 · Cleaning: find what is wrong, fix only what you can explain

Each press adds one row to the table: the problem, how you find it, and what you do. Then go through the code on the right, which is step 3 of notebook 0:

- `pd.to_datetime` turns date text into real dates, so they sort and subtract properly.
- `drop_duplicates("date")` keeps one row per day.
- `isna().sum()` counts the missing values in each column. Count before you fix.
- `interpolate(limit=2)` fills a gap with a straight line between its neighbours, but only gaps of at most two days. A long gap filled like this would be invented data.
- A negative rainfall is impossible, so it becomes unknown (`np.nan`) and is then filled with 0, the most common rainfall.
- A spike is a value far from the median of its neighbours (`rolling(15, center=True).median()`). One wrong value cannot move a median, so the median shows what the value should have been.

The note at the bottom is the habit that matters most: keep a list of every fix, so someone else can rebuild your dataset from the raw file.

### Slide 8 · Decide what goes in (X) and what comes out (y)

One row is one example. The blue columns are the inputs X, the orange one is the target y. The model learns the link from blue to orange.

Spend time on the second bullet. Only put in X what you will really have at the moment of prediction. Tomorrow's real temperature is not known today, but tomorrow's weather forecast is. A model trained with tomorrow's real temperature looks great on paper and fails in use. This mistake has a name, leakage, and it is the most common error in applied machine learning.

Then the shapes. A table is (examples, inputs), here (1094, 4). A time series is (examples, time steps, channels). Images are (examples, height, width, colours). The shape is the first thing to know about any dataset, because in a few slides it will choose the model.

### Slide 9 · Train, validation and test: three separate piles

The table gives the three piles and what each is for. Training data sets the knobs. Validation data is for our choices: when to stop training and how big a model to use. Test data is the final score and is looked at once.

Click *random split* on the animation. Red outlines appear on validation and test examples that sit right next to a training example. Say: "When data comes in time order, neighbouring examples are almost the same. Split at random and the test is mostly things the model has already seen. The score looks wonderful and means nothing. So we split by time: the past trains, the future tests."

The code is three slices of the arrays: the first 70 % trains, the next 15 % validates, the last 15 % tests. When there is no time order, for example one row per patient, a random split with `train_test_split` is fine. The rule is to split by whatever must be new when the model is used: time, patient, plant, sensor.

### Slide 10 · Put every input on a similar scale

Click the three buttons: raw values, log, standardised. The example column is a water flow of about 20 000 m³ a day.

Say: "Inside a model every input is multiplied by a weight and the results are added. A flow of 20 000 next to a temperature of 25 would drown it out. So we bring every input to the same scale, around zero with a spread of one. For quantities that change by factors, such as flows, we take the log first, so one big storm does not stretch the whole scale. Then we standardise: subtract the mean, divide by the standard deviation."

Now press the arrow for the most important rule of part 1. The mean and standard deviation come from the training data only (`StandardScaler().fit(X_train)`). Then exactly those numbers are used for validation, test and every later prediction. If you compute them from all the data, the test set has leaked into training. That is also why the scaler is saved with the model.

### Slide 11 · How the data gets into the model: tensors and batches

Press through the boxes. The whole training set, 765 rows of 4 inputs, goes into a `DataLoader`. It shuffles the rows and serves them 32 at a time. One batch is a 32 × 4 tensor, and the model turns it into 32 guesses at once.

- A tensor is an array PyTorch can compute with. It also remembers every calculation it was part of, which is how the gradients are found later.
- A batch is a few examples at a time: faster than one by one, and the small randomness helps.
- An epoch is one pass through all the batches. Training runs for many epochs.
- Shuffle the training batches, never the validation or test data.

The code: `TensorDataset` pairs each X row with its y row, `DataLoader` cuts them into shuffled batches, and `next(iter(train_dl))` takes one batch to look at its shape. Checking shapes like this catches most bugs before training starts.

### Slide 12 · What is a model?

Press **train** and let the line fit itself while you talk.

Say: "A model is a function with knobs. Here the function is a straight line, ŷ = w · x + b. The input x is what we know. The output ŷ is the model's guess of the answer y. The two knobs are w, the slope, and b, the offset. Those two numbers are the model's parameters. Training changes them and nothing else.

How wrong is the line? Each red line is one mistake. Square each mistake, take the average, and you have one number called the loss. Training means turning each knob a little in the direction that makes the loss smaller, and doing that again and again. A neural network is exactly this with thousands of knobs and a bendier function."

#### The code: `lesson1_knobs.py`

Run it with `python workshop/examples/lesson1_knobs.py`. It is the smallest model there is, two knobs, trained twice: once by hand in numpy, once with PyTorch. First the data.

{{code:workshop/examples/lesson1_knobs.py|rng = np.random.default_rng(0)|y = 0.7 * x}}

- `default_rng(0)` makes a random number generator with seed 0, so everyone in the room gets the same numbers.
- `x` is 20 input values between 0 and 1 (already scaled).
- `y` is the true answer. We make it ourselves with a hidden rule, slope 0.7 and offset 0.15, plus a little noise. The model is never told 0.7 or 0.15. It has to find them. That is why we invent the data here: we know the right answer, so we can check the model got it.

Now training by hand, in plain numpy.

{{code:workshop/examples/lesson1_knobs.py|w, b = 0.0, 0.0|the hidden rule was}}

- `w, b = 0.0, 0.0`: the knobs start at zero. The model knows nothing yet.
- `lr = 0.5` is the learning rate, the size of each turn of the knobs.
- `for step in range(201)` repeats the four steps 201 times.
- **Step 1, guess.** `pred = w * x + b` runs the model on all 20 inputs at once.
- **Step 2, how wrong.** `loss = np.mean((pred - y) ** 2)` is the mean squared error, the average of the squared mistakes.
- **Step 3, which way.** The gradient says how the loss changes when you nudge each knob. For this loss, calculus gives `mean(2·(pred − y)·x)` for w and `mean(2·(pred − y))` for b. If the gradient for w is positive, turning w up makes things worse, so we turn it down.
- **Step 4, turn the knobs.** `w = w - lr * grad_w` steps against the gradient, downhill.

Part B does exactly the same with PyTorch, the way every real training script does it.

{{code:workshop/examples/lesson1_knobs.py|xt = torch.tensor|opt.step()}}

- `torch.nn.Linear(1, 1)` is a layer with one input and one output: one weight (our w) and one bias (our b).
- `torch.optim.SGD(model.parameters(), lr=0.5)` hands the knobs to an optimiser. SGD is plain gradient descent, the same rule as Part A.
- `opt.zero_grad()` clears the gradients from the last step. PyTorch adds gradients up by default, so forgetting this line is a classic bug.
- `loss.backward()` computes every gradient automatically. This replaces the two hand-written lines of Part A, and it works for a network of any size.
- `opt.step()` turns every knob.

The output, from the run that built this guide:

{{run:lesson1_knobs}}

Read it with the room. The loss falls from 0.31 to 0.0011, and w and b end near 0.67 and 0.17. They do not land exactly on 0.7 and 0.15 because of the noise we added: with 20 noisy points, this is the best line. PyTorch arrives at the same answer.

> Try this with the room: change `lr = 0.5` to `1.2` and run again. The loss grows instead of shrinking, because each step jumps over the bottom of the valley and lands higher on the other side. The gradient-descent slide shows the same thing as a picture.

### Slide 13 · Three kinds of learning: it depends on what answers you have

Three cards, one per kind. Press the arrow to bring in the second and third.

- **Supervised**: there is a right answer y for every x. If y is a number it is regression (tomorrow's demand); if y is a class it is classification (spam or not). The loss compares the guess with y.
- **Unsupervised**: there is no y. The model finds structure in X alone: groups of similar samples (clustering), a few numbers that summarise many columns (PCA), or samples unlike the rest (anomaly detection).
- **Reinforcement**: no right answers at all, only a score after acting. An agent tries actions and learns which ones pay off. That is part 3.

Last press: our demand example is supervised regression, because for every past day we know what the demand of the next day really was.

### Slide 14 · The shape of your data picks the kind of model

Say: "Start from the shapes, not from the fashionable model. If your input is a table, start with a linear model or decision trees, then a small network (an MLP). If it is a sequence, use a model that knows about time order: a 1-D convolution, a recurrent network, or a Transformer. For images, a convolutional network. The output decides the last layer and the loss: a number needs a linear last layer and squared or absolute error; a class needs softmax and cross-entropy; yes or no needs a sigmoid and binary cross-entropy."

Then the green note, which is the rule that saves the most time: always build a baseline with no learning first, then a linear model, and only then something bigger. Our example is a table, 4 numbers in and 1 out, so a linear model or a small MLP.

### Slide 15 · What happens inside a neural network, layer by layer

Press *next step* by hand, slowly. This slide is where most people finally understand what a network is.

Say: "Three inputs come in: tomorrow's temperature, tomorrow's rain and today's demand. Each neuron in the first layer multiplies every input by its own weight, adds them, adds its bias, and gets one number z. Then it applies ReLU: if z is negative it becomes zero, otherwise it stays. The next layer does the same with these results. The last neuron gives the guess for tomorrow's demand. Blue lines are positive weights, red are negative, and thicker lines are bigger weights. Training only changes those numbers."

A whole layer at once is h = ReLU(W·x + b), one matrix multiplication. That is why networks run fast on hardware built for matrices.

### Slide 16 · The loss says what "wrong" means

Toggle the curves. Squared error (MSE) punishes big mistakes hard and aims at the mean. Absolute error (MAE) is calmer about outliers and aims at the median. The pinball loss is lopsided: at τ = 0.9, guessing too low costs 9 times more than guessing too high, so the best answer is the value exceeded only 10 % of the time. Train three of these and you get a band. Cross-entropy is the loss for classes and yes-or-no questions. The code on the slide shows that in PyTorch each loss is one line, and the result is always one number.

### Slide 17 · Gradient descent: walk downhill on the loss

Try the three learning rates. At 0.02 the ball barely moves. At 0.15 it walks to the bottom. At 0.5 it bounces from side to side and flies off. This is what happens in `lesson1_knobs.py` with a learning rate of 1.2, drawn. In practice most code uses Adam, which adapts the step size for each knob, sometimes with a learning-rate schedule and gradient clipping.

### Slide 18 · What training code looks like

The circle animates the five stages: a batch, forward, loss, backward, update. Below it the two loss curves are drawn epoch by epoch. Training loss keeps falling. Validation loss falls, then starts to rise. That turning point is where the model starts to memorise, and that epoch is the one we keep.

#### The code, step by step

This is `train()` from notebook 0, step 8.

- `torch.optim.Adam(model.parameters(), lr=1e-2)` hands every knob to the optimiser. Adam is gradient descent that adapts the step size for each knob.
- `model.train()` switches the model into training mode. It matters for layers such as dropout, which behave differently while training.
- `for xb, yb in train_dl` takes one shuffled batch at a time.
- `pred = model(xb)` is the forward pass, the guess. `loss_fn(pred, yb)` turns the 32 mistakes into one number.
- `opt.zero_grad()` clears the old gradients, `loss.backward()` computes new ones for every knob, and `opt.step()` turns the knobs. These are the four steps of the two-knob model, unchanged.
- After each epoch, `model.eval()` and `torch.no_grad()` switch learning off, and the validation loss is measured.
- `if val < best` keeps a copy of the knobs from the best epoch. `copy.deepcopy` matters: without it you would keep a reference to knobs that go on changing.

### Slide 19 · Counting parameters, and comparing with your data

Type layer sizes or click the presets. A dense layer from n inputs to m outputs has n × m weights and m biases.

The notebook 0 model, 4 → 16 → 1, has (4·16 + 16) + (16·1 + 1) = 97 knobs for 765 training examples, about one knob for every eight examples. It cannot memorise much. Click *too big*: 4 → 1024 → 1024 → 1 has over a million knobs for the same 765 examples, so it can learn every example by heart. The digit-image preset shows why image models are big: 784 pixels in already means 100 000 knobs in the first layer. Too many knobs for the data is not forbidden, but it means you must watch the validation curve.

This is how the size of a model is decided in practice: the shapes fix the first and last layer, and the amount of data limits what goes in between. The next slide chooses between a few sizes properly.

### Slide 20 · Parameters are learned; settings are chosen by you

This answers the question "how do we decide how big the model is, and how fast it learns?"

The table on the left: parameters are every weight and bias, set by training on the training data. Settings, also called hyperparameters, are the choices made before training: the number of layers and units, the learning rate, the batch size, the number of epochs, regularisation. Training cannot choose them, so we do, and we judge them on the validation data, never on the test.

The code tries three layer sizes and two learning rates, six trainings in all, and records the validation loss of each. Press for the table. The middle size (16 units, learning rate 0.01, 97 knobs) wins. Four units are too few to fit the pattern; 64 start to fit the noise. Bigger is not automatically better. The chosen settings are then trained once more, and only then tested, once (next slide).

Good starting points to give the room: a small model, Adam with a learning rate between 10⁻³ and 10⁻², and keep the best validation epoch. Grow the model only while the validation loss keeps falling.

### Slide 21 · What testing code looks like

Go through the code line by line:

- `load_state_dict(best_state)` puts back the knobs from the best validation epoch.
- `model.eval()` and `with torch.no_grad()` make sure nothing is learned here.
- The predictions come out scaled, so `* y_std + y_mean` turns them back into thousands of litres. Report errors in real units: "44 thousand litres a day" means something to the town, "0.22" means nothing.
- The baseline is the simplest guess there is, tomorrow equals today (`X_test[:, 3:4]`, the `demand_today` column).

Press the arrow for the result: the model is off by 44.4 thousand litres a day, the baseline by 56.7. The model earns its place, by about 22 %. Note where this sits: after the settings were chosen. The test is always the very last look.

Then the rules on the right. Test with the same preparation as training, in real units, against a baseline, and once. Press for the last two: if you change the model after seeing the test score, the test has become validation. For numbers the usual measures are MAE, RMSE and R²; for classes, accuracy, precision and recall, and the confusion matrix.

### Slide 22 · One notebook, twelve steps: the skeleton of every ML notebook

The whole of part 1 on one slide, as the code of notebook 0, steps 0 to 12. Read the green comments down the left column and then the right. Say: "Every notebook you will meet, including the project's, is these blocks in this order, with more detail inside each. When a notebook confuses you, ask which of the twelve steps a cell belongs to."

### Slide 23 · Your turn: Notebook 0: the skeleton

Everyone opens `workshop/notebooks/00_ml_skeleton.ipynb` and runs it from the top with Shift + Enter. Then they change one thing at a time. What they should find:

- Without `demand_today` the test error grows a lot. Yesterday's demand is the strongest input.
- With 256 hidden units the validation loss is no better, often worse.
- A learning rate of 1.0 makes the loss jump about or explode.
- A random split gives a test score that looks better than the time split, and should not be believed: the test days now sit between training days.

Anyone whose setup fails can run `python workshop/examples/lesson1_knobs.py` in a terminal instead.

### Slide 24 · Check yourself

Ask each question to the room before you reveal the answer.

1. Which parts of ŷ = w·x + b does training change? Only w and b. The data, the shape of the function and the settings stay as we chose them.
2. Where do the mean and standard deviation for scaling come from? The training data only, and they are saved with the model.
3. Training loss falls and validation loss rises. Overfitting: keep the best epoch, use a smaller model or regularisation, or get more data.
4. Why look at the test set only once? Any decision made after looking at it turns it into validation data.

## Part 2. Our project: plant data to a forecaster

### Slide 25 · Our project: plant data to a forecaster

Say: "Now the same recipe on real research data. Every step from part 1 comes back, and each one turns out harder: the file is an Excel sheet with a trap in it, one error can only be found with physics, and the data is a time series, not a table."

### Slide 26 · What the project is about

Go through the five bullets in order. The table on the right matters more than it looks: the project contains four different kinds of machine learning, and the scholars will see all four today.

- Forecasting the inflow two hours ahead is **regression**: the answer is a number.
- Warning before ammonium breaks the legal limit is **classification**: the answer is yes or no.
- Predicting how the plant will respond is a **sequence model**: the answer depends on everything that happened before.
- Deciding how much air to blow is **reinforcement learning**: there is no right answer to copy, only a score afterwards.

### Slide 27 · The plant and its two control loops

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

### Slide 28 · Our data: the official BSM1 influent files

The curves draw themselves as if a sensor were logging them, and rows appear in the table on the right.

Say: "Our data is what flows into the plant: the flow rate and 13 concentrations, every 15 minutes, for 14 days. That is 1345 rows. There are three files, one per weather: dry, rain and storm. Look at the first week. It is the same in all three files. Keep that in mind, because if we split carelessly we will test on data we trained on."

The rain file has one long rain period around days 8.4 to 10.4. The storm file has two short storms, around day 8.8 and day 11, with flows up to 60 000 m³/d, more than three times the dry average.

### Slide 29 · Load it once, record where it came from, cache it

This is the project's loading code. Four habits, one per bullet.

- **Pin the version.** The download address contains a commit hash. A file behind a hash never changes, so anyone can rebuild the same dataset years later.
- **Keep only the data rows.** Spreadsheets have titles and units in them. The project keeps a row only if its first cell is a number.
- **Cache it.** Reading Excel is slow. The project saves the numbers as `.npy`, which loads in milliseconds.
- **Respect the licence.** These files are under the GPL, so the repository downloads them when needed instead of storing copies.

### Slide 30 · Cleaning our files, and a fix only physics can find

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

### Slide 31 · Put everything on one time grid

The sensors give a value every 15 minutes, but the controller acts every 45 seconds. Between two samples we keep using the last one, a zero-order hold. That is 20 control steps per sample. The animation shows the held values filling in step by step.

### Slide 32 · Download, read, check, fix

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

### Slide 33 · Collect and clean, cell by cell

Notebook 1 running inside the slide. Each press runs one cell and shows its real output. The panel on the right explains the idea behind the cell. Use this slide if the room is slow to get started: they can watch the first cells run before trying themselves.

### Slide 34 · Turn one long series into many (input, target) examples

The animation slides a window along the flow curve. The blue part is the input, the last 24 hours. The orange part is the target, the next 2 hours. Each position of the window is one example. Stack them and you get an array of shape (examples, 96 time steps, 3 channels).

The input always ends exactly where the target begins. The model only ever sees the past, as it will on the real plant.

The split, as in part 1, is by time and weather. Training uses the dry file and the rain file's second week, validation is the first storm, test the second storm. The model never sees a storm while it trains, on purpose: the test asks whether it copes with something new. Other parts of the project split differently: paper 2 learns on week 1 and tests on week 2, and the warning model uses separate simulated plants for training, validation and test.

### Slide 35 · Choose, split, scale, cut: cell by cell

Notebook 1, part C, running cell by cell.

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

### Slide 36 · Your turn: Notebook 1: collect, clean, prepare

Scholars open `workshop/notebooks/01_data_collect_clean_prepare.ipynb` and run parts A, B and C. Walk around the room. The usual problems: they skip a cell, or their kernel was started in the wrong folder. The menu item *Kernel, Restart Kernel and Run All Cells* fixes most of it. Anyone stuck can run the lesson 2 and lesson 3 scripts instead. Ask them to think about this while it runs: training has no storm, the test is a storm. What will happen?

### Slide 37 · Our shapes, and the models in the project

Apply the table from part 1 to our data. The input is a sequence, (N, 96, 3): 96 time steps of flow, ammonium and COD. The output is 8 numbers, the next 8 flows. A sequence in and numbers out means a model that knows time order, such as a 1-D convolution, with a linear last layer. And as in part 1: baselines first, then a linear model, then an MLP, then the CNN.

Then click through the project's six models on the right. Each card gives the shapes, the model, the loss and the reason. The last one is the published MAACC controller, which has only 10 to 14 knobs that learn. Remember that for part 3.

### Slide 38 · A causal convolution: the same small filter slides along time

A small filter of 3 weights slides along time. Each output looks at 3 inputs, and only from the past: padding is added on the left so nothing from the future gets in. With dilations 1, 2, 4, each layer looks twice as far back, so a few layers cover a whole day with very few weights. The project's forecaster sees 253 steps back, more than its 96-step input.

### Slide 39 · Shapes, baselines and models, cell by cell

Notebook 2, sections 1 to 6, running. The layer-by-layer cell is worth showing even if you ran the script.

#### The code: `lesson4_models.py`

Run it with `python workshop/examples/lesson4_models.py`. It reads the file `lesson3_prepare.py` wrote.

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

### Slide 40 · Training, overfitting and the test, cell by cell

Notebook 2, sections 5 to 9, running. The overfitting cell draws both curves side by side.

### Slide 41 · Train, validate, test once, save

The four steps in the lesson script, notebook 2, the project's early-warning training code and the project's forecaster loader.

#### The code: `lesson5_train_and_evaluate.py`

Run it with `python workshop/examples/lesson5_train_and_evaluate.py`. It takes about 5 seconds.

{{code:workshop/examples/lesson5_train_and_evaluate.py|def mlp(hidden):|return nn.Sequential(*layers, nn.Linear(n, 8))}}

`torch.manual_seed(0)` fixes the random starting weights. Without it every run starts from different knobs and ends with slightly different numbers, and scholars ask why theirs do not match the slide. With it, everyone gets the same result.

{{code:workshop/examples/lesson5_train_and_evaluate.py|def fit(model, epochs|return np.array(curve), best_epoch}}

- `DataLoader(..., batch_size=128, shuffle=True)` serves the training examples in shuffled batches of 128. One pass through all of them is an epoch.
- `AdamW` is the optimiser. `weight_decay` gently pulls every weight towards zero, a mild brake on memorising.
- Inside the batch loop are the four steps of part 1 in two lines: guess and loss in one, then `zero_grad`, `backward`, `step`.
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

### Slide 42 · A model is only good compared with something simple

The left side lists the baselines. The bar chart on the right shows the project's own lesson. Read the bars from the README: a published learning controller, rebuilt exactly, did nothing beyond its PID, and our own learned controller was matched by a carefully tuned PID with the daily ILC. That is a result, not a failure, and it belongs in the report.

### Slide 43 · Your turn: Notebook 2: choose and train

The longest hands-on session. Ask everyone to fill in the table on the slide: parameters and test error at 2 hours for each model. When most have finished, ask who had the best model, and point out that the linear model is usually at or near the top.

### Slide 44 · Check yourself

1. Why did every column of the Excel file come out as text? One row of names makes pandas read the whole column as text.
2. All the usual checks passed. Can the data still be wrong? Yes. S_I stays at 30 while rain doubles the flow. Only physics catches it.
3. Read the shape (1906, 96, 3). 1906 examples, 96 time steps each, 3 measurements per step.
4. Why did the simplest model win on the storm? About 1900 examples and no storm in training. More varied data helps more than a bigger model.

## Part 3. Reinforcement learning and our controller

### Slide 45 · Reinforcement learning and our controller

Goal: learn from a reward instead of right answers, then see the project itself: how it was built, the published methods we rebuilt, our controller and the latest results.

### Slide 46 · When there is no correct answer to copy

The loop animates: the agent sees the state, picks an action, the tank responds, a reward comes back.

Say: "For control there is no right answer to copy. Nobody can tell us the perfect amount of air for every minute. So the agent tries actions, the plant responds, and a reward scores what happened. The reward is where you write down what you want. Paper 2's reward is on the slide: pay for the air you use, and pay for ammonium above the limit of 4. That agent saved 9 % energy, and it broke the ammonium limit far more often than the PID. It did exactly what the reward allowed."

### Slide 47 · The parts of an actor-critic agent (DDPG, SAC)

The table lists the parts: the environment, an actor network that picks actions, a critic network that estimates the future reward of an action, a replay buffer of past experience, slow target copies of the networks, and exploration noise. The code on the right is one DDPG update. Point at the critic's loss: it is a squared error against a target, the same kind of regression as in part 1.

### Slide 48 · Reinforcement learning, cell by cell

Notebook 3 running, including the learning curve and the comparison on the unseen rain day.

#### The physics of the toy tank

The script `lesson6_rl.py` and notebook 3 use one aerated tank, small enough to train in seconds. Its oxygen follows a mass balance:

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

### Slide 49 · Your turn: Notebook 3: reinforcement learning in brief

The question on the slide is the best discussion of the day: change the energy weight in the reward and watch the oxygen. Ask where the behaviour flips. In our runs the agent still holds the oxygen at a weight of 3, starts cutting the air at 4, and gives up on the oxygen at 5.

### Slide 50 · How the project was built: one script per step

Press the right arrow and walk through the boxes in order. Each box is one script in `experiments/`, and the panel shows the command that runs it and the file it writes. The story to tell:

1. Rebuild the official plant and its data.
2. Rebuild three published controllers exactly as their papers print them, and check their claims. Two of the three did not work as printed.
3. Build our own controller and two strong baselines to compare it with.
4. Test everything on data nothing was tuned on, twice, with the test plan committed to git before each test was run.
5. Audit every number. 76 of 76 checks pass.

The point to land: research code is a chain of small scripts, each writing a result file that the next one reads, so every number in the report can be traced back to a command.

### Slide 51 · Reward, DDPG update and SAC-IL update, in the project's code

A code tour in three tabs. The paper 2 reward, two lines. One DDPG update: replay batch, Bellman target, critic regression, actor step, soft target update. One SAC-IL update: a quarter of each batch comes from the expert's demonstrations, and the imitation term never fades below a floor.

{{code:src/wwtp/rl/ddpg.py|# eq. (9): y = r + gamma|self.opt_actor.step()}}

- `y = r + γ · (1 − d) · Q_target(s′, μ_target(s′))` is the Bellman target, made with the slow target networks so it does not jump every step. `(1 − d)` drops the future at the end of an episode.
- `loss_c` is the squared error between the critic's estimate and that target. This is the same regression as in part 1.
- `loss_a` is minus the critic's value of the actor's action. Minimising it means the actor climbs the critic. The small penalty on `pre` keeps the `tanh` output from getting stuck at a limit.

### Slide 52 · Start from the expert, then improve, behind a safety gate

The project's controller in four boxes. The expert PID is run on dry week 1 and its moves recorded. A network copies it (behaviour cloning, which is supervised learning). Soft actor-critic improves on that copy using the reward. A safety gate switches the learned part off whenever the inputs leave the range seen in training. The code on the left is the copying step, a plain regression.

### Slide 53 · Papers 1 and 3 exactly as printed, and why they don't learn

Two code tours. Paper 1's MAACC is a small adaptive-critic controller added on top of a PID.

{{code:src/wwtp/maacc/exact.py|def step(self, z: np.ndarray) -> float:|return a}}

- The action network `alpha @ bipolar(eta * zi)` gives a change of input, which is added up into `a`. Because `alpha` starts at zero, the controller starts exactly as its PID.
- The critic estimates the value `q` and the utility `U` scores the moment, with the weights printed in the paper's Table I.
- The update of `alpha` is the learning rate times `q` times the slope of the critic. With the printed weights that product is about 10⁻⁴, so `alpha` hardly moves. In 80 of 80 runs, MAACC equals its PID to within 1 %. The code is correct. The method, as printed, cannot learn on this plant.

Paper 3's IOQL repeats 50 critic updates and 60 actor updates on each single sample. That many steps on one point makes it unstable: it either stays at its PID or the oxygen collapses.

### Slide 54 · ILC, the control step, the safety gate and the training script

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

### Slide 55 · The second held-out test: two fixes, one open problem

This slide holds the project's latest results (`docs/PAPER1_HELDOUT2.md`). The plan for this test was committed to git before any of its 432 runs.

Held-out test 1 had found two flaws. Under overload the ILC made things worse, and every high-gain controller failed with the official sensor noise. Two fixes were made:

1. **ILC anti-windup**, the code above.
2. **A robust PID**: a 0.75-minute filter on the oxygen reading and half the gains, chosen on dry week 1 with spare noise seeds that no test uses.

They were then tested on new BSM2 influent windows, with and without a primary clarifier, and with new sensor-noise seeds. Read the table on the slide:

- Anti-windup works on new data. Under overload it removes the ILC's harm, 39 % better, and costs nothing in normal operation.
- The robust PID never failed and is 83 % better than the paper's PID on new influent, with about the same blower movement as the paper's PID (522 against 501).
- With the official sensor noise, 0.25 g/m³ on a set point of 2 mg/L, nothing built in the project beats the paper's own gentle PID. That is the open problem. Next ideas are a Kalman filter on the oxygen reading, or switching to lower gains when the noise is high.

Say: "This is what honest research looks like. Two fixes worked and are reported. One problem is still open and is reported too. A test that only uses clean data, as paper 1 did, would have hidden all of this."

### Slide 56 · Check yourself

1. What is the difference between a reward and a label? A label is the right answer. A reward only scores what happened.
2. What did the agent do when air was made very expensive? It switched the air off.
3. What does the safety gate do? Switches the learned part off on inputs outside the training range.
4. What did held-out test 2 show? Anti-windup fixed overload, the robust PID never failed, and under sensor noise nothing beat the paper's PID.

## Closing

### Slide 57 · Questions learners ask, with short answers

Eight questions with short answers, collected by going through the lesson as a beginner. Open the ones the room cares about.

### Slide 58 · What this project teaches about doing ML research

End on the eight habits. Look at the data first. Let the shapes choose the model. Split honestly. Beat simple baselines. Reproduce published work before claiming to improve it. Use several seeds. Write the test down before you run it. Save the model with everything it needs. Then take questions.

## After the session

Everything is in the workshop folder. The scholars can rerun any step with one command from `workshop/examples/`, and notebook 0 is the skeleton to copy for their own projects. The notebooks are also web pages with every output. The project's own documents (`docs/PAPER1_STUDY.md`, `docs/PAPER1_HELDOUT2.md`, `docs/PAPER1_AUDIT.md`) take them as deep as they want to go.

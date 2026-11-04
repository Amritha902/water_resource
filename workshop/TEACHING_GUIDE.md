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

```python
rng = np.random.default_rng(0)
x = rng.uniform(0, 1, 20)
y = 0.7 * x + 0.15 + 0.05 * rng.normal(size=20)     # the "true" rule, hidden from the model
```
<p class="src">workshop/examples/lesson1_knobs.py, lines 14–16</p>


- `default_rng(0)` makes a random number generator with seed 0, so everyone in the room gets the same numbers.
- `x` is 20 input values between 0 and 1 (already scaled).
- `y` is the true answer. We make it ourselves with a hidden rule, slope 0.7 and offset 0.15, plus a little noise. The model is never told 0.7 or 0.15. It has to find them. That is why we invent the data here: we know the right answer, so we can check the model got it.

Now training by hand, in plain numpy.

```python
w, b = 0.0, 0.0          # THE KNOBS (parameters). Training changes only these two numbers.
lr = 0.5                 # LEARNING RATE: how far we turn the knobs each step

print("Part A: by hand")
for step in range(201):
    pred = w * x + b                          # 1 GUESS: run the model on the data (forward)
    loss = np.mean((pred - y) ** 2)           # 2 HOW WRONG: one number, the mean squared error
    if step in (0, 1, 2, 5, 10, 50, 200):
        print(f"  step {step:3d}   w = {w:6.3f}   b = {b:6.3f}   loss = {loss:.5f}")
    # 3 WHICH WAY: the gradient = how the loss changes when a knob is nudged.
    #   loss = mean((w·x + b − y)²); by the chain rule  d loss/dw = mean(2·(pred − y)·x),  d loss/db = mean(2·(pred − y)).
    #   PyTorch computes this for you with loss.backward() (Part B).
    grad_w = np.mean(2 * (pred - y) * x)      # 3 WHICH WAY: the gradient, for each knob
    grad_b = np.mean(2 * (pred - y))
    w = w - lr * grad_w                       # 4 TURN THE KNOBS a little, downhill
    b = b - lr * grad_b
print("  the hidden rule was w = 0.7, b = 0.15\n")
```
<p class="src">workshop/examples/lesson1_knobs.py, lines 19–35</p>


- `w, b = 0.0, 0.0`: the knobs start at zero. The model knows nothing yet.
- `lr = 0.5` is the learning rate, the size of each turn of the knobs.
- `for step in range(201)` repeats the four steps 201 times.
- **Step 1, guess.** `pred = w * x + b` runs the model on all 20 inputs at once.
- **Step 2, how wrong.** `loss = np.mean((pred - y) ** 2)` is the mean squared error, the average of the squared mistakes.
- **Step 3, which way.** The gradient says how the loss changes when you nudge each knob. For this loss, calculus gives `mean(2·(pred − y)·x)` for w and `mean(2·(pred − y))` for b. If the gradient for w is positive, turning w up makes things worse, so we turn it down.
- **Step 4, turn the knobs.** `w = w - lr * grad_w` steps against the gradient, downhill.

Part B does exactly the same with PyTorch, the way every real training script does it.

```python
xt = torch.tensor(x, dtype=torch.float32)[:, None]
yt = torch.tensor(y, dtype=torch.float32)[:, None]
torch.manual_seed(0)
model = torch.nn.Linear(1, 1)                 # THE KNOBS: model.weight (= w) and model.bias (= b)
opt = torch.optim.SGD(model.parameters(), lr=0.5)

print("Part B: PyTorch")
for step in range(201):
    pred = model(xt)                                  # 1 guess          (notebook 2, cell 9, line 19)
    loss = torch.nn.functional.mse_loss(pred, yt)     # 2 how wrong      (notebook 2, cell 9, line 20)
    if step in (0, 50, 200):
        print(f"  step {step:3d}   w = {model.weight.item():6.3f}   b = {model.bias.item():6.3f}   loss = {loss.item():.5f}")
    opt.zero_grad()
    loss.backward()                                   # 3 which way      (notebook 2, cell 9, line 22)
    opt.step()                                        # 4 turn the knobs (notebook 2, cell 9, line 24)
```
<p class="src">workshop/examples/lesson1_knobs.py, lines 38–52</p>


- `torch.nn.Linear(1, 1)` is a layer with one input and one output: one weight (our w) and one bias (our b).
- `torch.optim.SGD(model.parameters(), lr=0.5)` hands the knobs to an optimiser. SGD is plain gradient descent, the same rule as Part A.
- `opt.zero_grad()` clears the gradients from the last step. PyTorch adds gradients up by default, so forgetting this line is a classic bug.
- `loss.backward()` computes every gradient automatically. This replaces the two hand-written lines of Part A, and it works for a network of any size.
- `opt.step()` turns every knob.

The output, from the run that built this guide:

```text
Part A: by hand
  step   0   w =  0.000   b =  0.000   loss = 0.30951
  step   1   w =  0.332   b =  0.512   loss = 0.04186
  step   2   w =  0.280   b =  0.342   loss = 0.01765
  step   5   w =  0.382   b =  0.330   loss = 0.00996
  step  10   w =  0.480   b =  0.274   loss = 0.00495
  step  50   w =  0.664   b =  0.172   loss = 0.00113
  step 200   w =  0.671   b =  0.168   loss = 0.00112
  the hidden rule was w = 0.7, b = 0.15

Part B: PyTorch
  step   0   w = -0.007   b =  0.536   loss = 0.04909
  step  50   w =  0.661   b =  0.174   loss = 0.00113
  step 200   w =  0.671   b =  0.168   loss = 0.00112
```
<p class="src">Output of <code>python workshop/examples/lesson1_knobs.py</code>, from the run that built this guide</p>


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

```python
# ── 1 DOWNLOAD: pinned to one commit, so it is the same file for everyone, forever
path = DATA / "Inf_rain_2006.xlsx"
if not path.exists():
    urllib.request.urlretrieve(URL + path.name, path)
dry_path = DATA / "Inf_dry_2006.xlsx"
if not dry_path.exists():
    urllib.request.urlretrieve(URL + dry_path.name, dry_path)
```
<p class="src">workshop/examples/lesson2_load_and_clean.py, lines 24–30</p>


- `DATA / "Inf_rain_2006.xlsx"` builds a path to the data folder. `Path` objects join folders with `/`, which also works on Windows.
- `if not path.exists():` downloads only the first time. After that the file on disk is used.
- `urlretrieve` fetches the file from the pinned address in `URL`.

```python
# ── 2 READ: first, carelessly
raw = pd.read_excel(path, header=None)
print("2 READ   raw shape", raw.shape, "  first row:", list(raw.iloc[0, :4]), "...")
print("         column types:", {str(k): v for k, v in raw.dtypes.value_counts().items()}, " ← all text, because row 0 is names")
#    then properly: row 0 is the header; give the columns our names; make them numbers
d = pd.read_excel(path, header=0)
d.columns = COLS
d = d.astype(float)
print("         after reading properly:", d.shape, " types:", {str(k): v for k, v in d.dtypes.value_counts().items()})
```
<p class="src">workshop/examples/lesson2_load_and_clean.py, lines 32–40</p>


- `pd.read_excel(path, header=None)` reads the sheet without saying which row holds the column names. The first row of this file holds names like `t`, `Si`, `Ss`. Because one row is text, pandas treats every column as text (type `object`). Nothing crashes, which is what makes this trap dangerous.
- `header=0` says row 0 is the header. Now the 1345 rows below it are data.
- `d.columns = COLS` gives the columns names we choose and understand.
- `d.astype(float)` makes every column a number, and fails loudly if something is not a number. Loud failures are good.

```python
# ── 3 CHECK: the standard questions, every dataset, every time
step_min = np.unique(np.round(np.diff(d.t) * 1440, 3))
print("3 CHECK  time step [min]:", step_min,
      "  missing values:", int(d.isna().sum().sum()),
      "  duplicate times:", int(d.t.duplicated().sum()),
      "  negative values:", int((d.drop(columns="t") < 0).sum().sum()))
print("         S_I values in the whole file:", d.S_I.unique(), " ← constant, even when rain doubles the flow?")
```
<p class="src">workshop/examples/lesson2_load_and_clean.py, lines 42–48</p>


- `np.diff(d.t) * 1440` is the gap between consecutive times, converted from days to minutes (1440 minutes per day). We expect exactly one value, 15.
- `d.isna().sum().sum()` counts missing values in the whole table.
- `d.t.duplicated().sum()` counts repeated timestamps.
- `(d.drop(columns="t") < 0).sum().sum()` counts negative values. Concentrations and flows cannot be negative.
- The last line prints every distinct value of S_I. There is only one, 30, which is the physics problem.

```python
# ── 4 FIX (a) a broken sensor: damage a copy, then repair it
x = d.copy()
x.loc[100:102, "S_NH"] = np.nan                          # a 45-min gap
x.loc[300, "S_NH"] *= 10                                 # a spike
med = x.S_NH.rolling(9, center=True, min_periods=3).median()
spike = (x.S_NH - med).abs() > 0.5 * med                 # far from its neighbours → wrong
x.loc[spike, "S_NH"] = np.nan
x["S_NH"] = x.S_NH.interpolate(limit_area="inside")      # fill short gaps with a straight line
print("4 FIX    spikes found:", int(spike.sum()), "  largest error after repair:",
      round(float((x.S_NH - d.S_NH).abs().max()), 3), "g/m3")
```
<p class="src">workshop/examples/lesson2_load_and_clean.py, lines 50–59</p>


- `x = d.copy()` works on a copy, so we can compare the repair with the original.
- We damage the copy on purpose: three readings go missing and one becomes ten times too big.
- `rolling(9, center=True).median()` is the median of each reading and its 8 neighbours, 4 on each side. One spike cannot move a median, so the median shows what the value should be.
- `spike = (x.S_NH - med).abs() > 0.5 * med` flags a reading more than 50 % away from its neighbourhood.
- `interpolate(limit_area="inside")` fills the holes with straight lines between the values on both sides.

One caution for the room: `center=True` uses readings from after the moment being checked. That is fine for cleaning a stored file. A controller running live cannot see the future and must use only past values.

```python
# ── 4 FIX (b) the physics fix of the project: rain dilutes the inert COD   (plant.py, line 89)
dry = pd.read_excel(dry_path, header=0)
dry.columns = COLS
d["S_I"] = 30.0 * dry.Q.astype(float).values / d.Q.values
print("         S_I after the dilution fix: min", round(d.S_I.min(), 1), " max", round(d.S_I.max(), 1), "g/m3")
```
<p class="src">workshop/examples/lesson2_load_and_clean.py, lines 61–65</p>


`30.0 * dry.Q / d.Q` is the dilution rule: when the flow is twice the dry-weather flow, S_I is half of 30.

```text
2 READ   raw shape (1346, 15)   first row: ['t', 'Si', 'Ss', 'Xi'] ...
         column types: {'object': 15}  ← all text, because row 0 is names
         after reading properly: (1345, 15)  types: {'float64': 15}
3 CHECK  time step [min]: [15.]   missing values: 0   duplicate times: 0   negative values: 0
         S_I values in the whole file: [30.]  ← constant, even when rain doubles the flow?
4 FIX    spikes found: 1   largest error after repair: 1.548 g/m3
         S_I after the dilution fix: min 10.0  max 30.0 g/m3
```
<p class="src">Output of <code>python workshop/examples/lesson2_load_and_clean.py</code>, from the run that built this guide</p>


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

```python
# ── 1 CHOOSE: inputs a real plant measures at the inlet; target = the flow in the next 2 hours
CHANNELS = ["Q", "S_NH", "S_S"]                             # as forecast/dataset.py: CHANNELS
LOOKBACK, HORIZON = 96, 8                                   # 24 h in, 2 h out (15-min samples)
```
<p class="src">workshop/examples/lesson3_prepare.py, lines 33–35</p>


- `CHANNELS` lists the inputs: flow, ammonium and readily biodegradable COD. These are things a real plant can measure at the inlet. We do not give the model anything a plant could not know.
- `LOOKBACK = 96`: 96 samples of 15 minutes is 24 hours of history. One whole day, so the model can see the daily pattern.
- `HORIZON = 8`: 8 samples is 2 hours ahead.

```python
# ── 2 SPLIT by the time of the TARGET: train = past and calm, val = storm 1, test = storm 2
SPLITS = {"train": [("dry", 1.0, 14.0), ("rain", 7.0, 14.0)],
          "val": [("storm", 7.0, 10.25)],
          "test": [("storm", 10.25, 14.0)]}
```
<p class="src">workshop/examples/lesson3_prepare.py, lines 38–41</p>


The split is by the time of the target, in days. Training uses the dry file from day 1 (day 0 only serves as history) and the rain file's second week. Validation is storm 1, test is storm 2. The model never sees a storm while it trains. That is on purpose: the test asks whether it can cope with something new.

```python
# ── 3 SCALE: log, then (x − mean) / std, with mean and std from the training rows only
train_rows = np.concatenate([df[w][(df[w].t >= lo - 1) & (df[w].t < hi)][CHANNELS].values for w, lo, hi in SPLITS["train"]])
MEAN, STD = np.log(train_rows).mean(0), np.log(train_rows).std(0)
leaky = np.log(np.concatenate([df[w][CHANNELS].values for w in FILES])).mean(0)
print("3 SCALE   mean of log, training rows only:", MEAN.round(3), "  with the test leaking in:", leaky.round(3))


def scale(x):
    return (np.log(x) - MEAN) / STD
```
<p class="src">workshop/examples/lesson3_prepare.py, lines 44–52</p>


- `train_rows` gathers every raw sample the training windows touch, including their 24 hours of history (`lo - 1`).
- `MEAN, STD` are computed from the log of those rows only.
- `leaky` is what the mean would be if the test data were included. We print it only to show the difference. The numbers move by about 0.02, which looks small, but it is information from the future that should not be there.
- `scale(x)` applies the same transform anywhere: log, minus the training mean, divided by the training standard deviation.

```python
# ── 4 WINDOWS: past 96 steps → next 8 flows; the input always ends before the target starts
def windows(weather, t_lo, t_hi):
    z, t = scale(df[weather][CHANNELS].values).astype(np.float32), df[weather].t.values
    X, Y, T = [], [], []
    for s in range(len(z) - LOOKBACK - HORIZON + 1):
        first = s + LOOKBACK                                # first target row
        if t[first] < t_lo - 1e-9 or t[first + HORIZON - 1] >= t_hi - 1e-9:
            continue
        X.append(z[s:first]); Y.append(z[first:first + HORIZON, 0]); T.append(t[first])
    return np.array(X), np.array(Y), np.array(T)
```
<p class="src">workshop/examples/lesson3_prepare.py, lines 55–64</p>


- For each start position `s`, `first = s + LOOKBACK` is the row where the target begins.
- The `if` skips windows whose target lies outside this split, so no window crosses from training into test.
- `z[s:first]` is the input: 96 rows, 3 channels. `z[first:first + HORIZON, 0]` is the target: the next 8 values of channel 0, the flow.

```text
1 CHOOSE  inputs: ['Q', 'S_NH', 'S_S']   one example: (96, 3) → (8,)
2 SPLIT    {'train': [('dry', 1.0, 14.0), ('rain', 7.0, 14.0)], 'val': [('storm', 7.0, 10.25)], 'test': [('storm', 10.25, 14.0)]}
3 SCALE   mean of log, training rows only: [9.846 3.312 4.071]   with the test leaking in: [9.834 3.329 4.089]
4 WINDOWS train X (1906, 96, 3)  y (1906, 8)
4 WINDOWS val   X (305, 96, 3)  y (305, 8)
4 WINDOWS test  X (353, 96, 3)  y (353, 8)
          a random split would put 98% of test windows right next to a training window (sharing 95 of 96 samples)
          saved workshop/data/prepared.npz (windows + the scaler, together)
```
<p class="src">Output of <code>python workshop/examples/lesson3_prepare.py</code>, from the run that built this guide</p>


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

```python
d = np.load(Path(__file__).resolve().parents[1] / "data" / "prepared.npz")
X = {s: d[f"X_{s}"] for s in ("train", "val")}
Y = {s: d[f"y_{s}"] for s in ("train", "val")}
MEAN, STD = d["mean"], d["std"]
flow = lambda z: np.exp(np.asarray(z) * STD[0] + MEAN[0])                 # back to m³/d
mae = lambda p: np.abs(flow(p) - flow(Y["val"])).mean(0)                 # error per horizon, m³/d
count = lambda m: sum(p.numel() for p in m.parameters())
```
<p class="src">workshop/examples/lesson4_models.py, lines 20–26</p>


- `np.load` opens the prepared file. `X` and `Y` hold the training and validation arrays.
- `flow(z)` undoes the scaling: multiply by the standard deviation, add the mean, then `exp` undoes the log. We always report errors in m³/d, the units an engineer understands.
- `mae(p)` is the mean absolute error for each of the 8 horizons.
- `count(m)` adds up the size of every parameter tensor: the number of knobs.

```python
# ── 2 BASELINES: no learning at all
for name, pred in (("persistence (= now)", np.repeat(X["val"][:, -1:, 0], 8, 1)),
                   ("same time yesterday", X["val"][:, :8, 0])):
    e = mae(pred)
    print(f"2 BASELINE   {name:22s} error 15 min {e[0]:6.0f}   2 h {e[-1]:6.0f} m³/d   knobs: 0")
```
<p class="src">workshop/examples/lesson4_models.py, lines 31–35</p>


Two rules with no learning. *Persistence* says the next 2 hours equal the last value: `X["val"][:, -1:, 0]` is the last flow of each window, repeated 8 times. *Same time yesterday* uses the fact that the input is exactly one day long, so the value 24 hours before each target is `X["val"][:, :8, 0]`.

```python
# ── 3 LINEAR: flatten the window into 288 numbers; ŷ = W·x + b
lin = Ridge(alpha=1.0).fit(X["train"].reshape(len(X["train"]), -1), Y["train"])
e = mae(lin.predict(X["val"].reshape(len(X["val"]), -1)))
print(f"3 LINEAR     knobs {lin.coef_.size + lin.intercept_.size}  (= 288 × 8 + 8)   error 15 min {e[0]:6.0f}   2 h {e[-1]:6.0f} m³/d")
```
<p class="src">workshop/examples/lesson4_models.py, lines 37–40</p>


`reshape(len(...), -1)` flattens each 96 × 3 window into one row of 288 numbers, a table. `Ridge` is linear regression with a small penalty on large weights, which keeps it stable. It has 288 × 8 weights and 8 biases.

```python
# ── 4 MLP: the same, with two hidden layers and ReLU in between
torch.manual_seed(0)
mlp = nn.Sequential(nn.Flatten(), nn.Linear(288, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))
print(f"4 MLP        knobs {count(mlp)}  = (288·64 + 64) + (64·64 + 64) + (64·8 + 8)")
h = torch.from_numpy(X["val"][:1])
print(f"             {'input':26s} {tuple(h.shape)}")
for layer in mlp:
    h = layer(h)
    extra = f"   {(h == 0).float().mean():.0%} switched off by ReLU" if isinstance(layer, nn.ReLU) else ""
    print(f"             {str(layer)[:26]:26s} {tuple(h.shape)}{extra}")
```
<p class="src">workshop/examples/lesson4_models.py, lines 42–51</p>


- `nn.Sequential` stacks layers in order: flatten, a linear layer to 64, ReLU, another to 64, ReLU, and a final linear layer to 8 outputs.
- The loop passes one example through the layers one at a time and prints the shape after each. It also counts how many values ReLU set to zero. With random starting weights about half are switched off.

```python
# ── 5 CNN: a small filter slides along time; padding only on the left = causal
class CausalBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.pad, self.conv = 2 * dilation, nn.Conv1d(ch, ch, 3, dilation=dilation)

    def forward(self, x):
        return x + F.gelu(self.conv(F.pad(x, (self.pad, 0))))           # pad LEFT only: no future


cnn = nn.Sequential(nn.Conv1d(3, 24, 1), *[CausalBlock(24, d) for d in (1, 2, 4, 8, 16, 32)])
head = nn.Linear(24, 8)
print(f"5 CNN        knobs {count(cnn) + count(head)}  (fewer than the MLP, for the same input)")
x = torch.from_numpy(X["val"][:1]).transpose(1, 2)                      # (batch, channels, time)
out = cnn(x)
print(f"             (1, 3, 96) → convolutions → {tuple(out.shape)} → last step {tuple(out[:, :, -1].shape)} → head {tuple(head(out[:, :, -1]).shape)}")
x2 = x.clone(); x2[:, :, 50] += 5.0                                     # change one moment in the middle
changed = (cnn(x2) - out).abs().sum(1)[0] > 1e-6
print(f"             change time step 50 → outputs change from step {int(changed.nonzero()[0])} on, never before it (causal)")
```
<p class="src">workshop/examples/lesson4_models.py, lines 54–72</p>


- `CausalBlock`: `F.pad(x, (self.pad, 0))` adds `pad` zeros at the start of the time axis and none at the end. That is what makes it causal. `x + ...` adds the input back (a residual connection), which helps deep stacks train.
- The network is a 1 × 1 convolution from 3 to 24 channels, then six blocks with dilations 1 to 32.
- The test at the end changes one moment, time step 50, and checks which outputs change. Only outputs from step 50 onwards change. That proves no output looked into the future.

```text
1 SHAPES     X (1906, 96, 3) → y (1906, 8)   (examples, time steps, channels) → (examples, 8 future flows)
2 BASELINE   persistence (= now)    error 15 min   1515   2 h   5749 m³/d   knobs: 0
2 BASELINE   same time yesterday    error 15 min   4148   2 h   4128 m³/d   knobs: 0
3 LINEAR     knobs 2312  (= 288 × 8 + 8)   error 15 min   1078   2 h   2929 m³/d
4 MLP        knobs 23176  = (288·64 + 64) + (64·64 + 64) + (64·8 + 8)
             input                      (1, 96, 3)
             Flatten(start_dim=1, end_d (1, 288)
             Linear(in_features=288, ou (1, 64)
             ReLU()                     (1, 64)   52% switched off by ReLU
             Linear(in_features=64, out (1, 64)
             ReLU()                     (1, 64)   52% switched off by ReLU
             Linear(in_features=64, out (1, 8)
5 CNN        knobs 10808  (fewer than the MLP, for the same input)
             (1, 3, 96) → convolutions → (1, 24, 96) → last step (1, 24) → head (1, 8)
             change time step 50 → outputs change from step 50 on, never before it (causal)
```
<p class="src">Output of <code>python workshop/examples/lesson4_models.py</code>, from the run that built this guide</p>


The numbers to point at: persistence is good at 15 minutes and poor at 2 hours. Yesterday is the same at every horizon. The linear model beats both at every horizon. The CNN has fewer knobs than the MLP (10 808 against 23 176) for the same input, because one small filter is reused at every time step.

### Slide 40 · Training, overfitting and the test, cell by cell

Notebook 2, sections 5 to 9, running. The overfitting cell draws both curves side by side.

### Slide 41 · Train, validate, test once, save

The four steps in the lesson script, notebook 2, the project's early-warning training code and the project's forecaster loader.

#### The code: `lesson5_train_and_evaluate.py`

Run it with `python workshop/examples/lesson5_train_and_evaluate.py`. It takes about 5 seconds.

```python
def mlp(hidden):
    torch.manual_seed(0)                                        # same starting knobs every run: same numbers for everyone
    layers, n = [nn.Flatten()], 288
    for h in hidden:
        layers += [nn.Linear(n, h), nn.ReLU()]; n = h
    return nn.Sequential(*layers, nn.Linear(n, 8))
```
<p class="src">workshop/examples/lesson5_train_and_evaluate.py, lines 30–35</p>


`torch.manual_seed(0)` fixes the random starting weights. Without it every run starts from different knobs and ends with slightly different numbers, and scholars ask why theirs do not match the slide. With it, everyone gets the same result.

```python
def fit(model, epochs, n_train=None, wd=1e-4, lr=3e-3):
    torch.manual_seed(0)
    xt, yt = X["train"][:n_train], Y["train"][:n_train]
    loader = DataLoader(TensorDataset(xt, yt), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, curve = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:                                   # ── 1 TRAIN: the four steps, batch by batch
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():                                   # ── 2 VALIDATE: no learning here
            tr = F.mse_loss(model(xt), yt).item()
            va = F.mse_loss(model(X["val"]), Y["val"]).item()
        curve.append((tr, va))
        if va < best:                                           #    keep the best epoch
            best, best_state, best_epoch = va, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return np.array(curve), best_epoch
```
<p class="src">workshop/examples/lesson5_train_and_evaluate.py, lines 38–57</p>


- `DataLoader(..., batch_size=128, shuffle=True)` serves the training examples in shuffled batches of 128. One pass through all of them is an epoch.
- `AdamW` is the optimiser. `weight_decay` gently pulls every weight towards zero, a mild brake on memorising.
- Inside the batch loop are the four steps of part 1 in two lines: guess and loss in one, then `zero_grad`, `backward`, `step`.
- `model.eval()` and `torch.no_grad()` switch learning off while we measure. We measure both the training and the validation loss after every epoch.
- `if va < best:` keeps a copy of the weights from the epoch with the lowest validation loss. `copy.deepcopy` matters: without it we would keep a reference to weights that go on changing.
- At the end `load_state_dict(best_state)` puts the best epoch back.

```python
big = mlp((512, 512))
curve_big, be_big = fit(big, 80, n_train=200, wd=0.0, lr=1e-3)
print(f"    OVERFITTING     415k knobs on 200 examples: train loss {curve_big[0, 0]:.3f} → {curve_big[-1, 0]:.4f}, "
      f"validation {curve_big[be_big, 1]:.3f} at epoch {be_big} → {curve_big[-1, 1]:.3f} at the end  (it memorised)")
```
<p class="src">workshop/examples/lesson5_train_and_evaluate.py, lines 64–67</p>


The overfitting experiment: a network with 415 000 knobs trained on only 200 examples, with no weight decay.

```python
# ── 3 TEST: once. Persistence, linear, MLP on storm 2
lin = Ridge(alpha=1.0).fit(X["train"].reshape(len(X["train"]), -1).numpy(), Y["train"].numpy())
with torch.no_grad():
    tests = {"persistence": X["test"][:, -1:, 0].repeat(1, 8).numpy(),
             "linear": lin.predict(X["test"].reshape(len(X["test"]), -1).numpy()),
             "MLP": net(X["test"]).numpy()}
print("3 TEST (once)       error 2 h ahead on storm 2: " + "   ".join(f"{k} {mae2h(v, 'test'):.0f}" for k, v in tests.items()) + "  m³/d")
```
<p class="src">workshop/examples/lesson5_train_and_evaluate.py, lines 69–75</p>


The test set, used once, at the end, for all three models. Nothing about the models is changed after this line.

```python
# ── 4 SAVE everything the model needs; then use it like the plant would
torch.save({"state_dict": net.state_dict(), "mean": MEAN, "std": STD, "hidden": (64, 64),
            "channels": ["Q", "S_NH", "S_S"]}, DATA / "lesson5_mlp.pt")
blob = torch.load(DATA / "lesson5_mlp.pt", weights_only=False)
model = mlp(blob["hidden"]); model.load_state_dict(blob["state_dict"]); model.eval()
for i, when in ((len(d["X_test"]) - 1, "calm, end of test"), (100, "during storm 2")):
    raw = np.exp(d["X_test"][i] * STD + MEAN)                           # 24 h of raw sensor values
    z = (np.log(raw) - blob["mean"]) / blob["std"]                      # the SAVED scaler, not a new one
    with torch.no_grad():
        out = flow(model(torch.from_numpy(z[None].astype(np.float32)))[0].numpy())
    print(f"4 SAVE + RELOAD     {when:18s} forecast in 2 h: {out[-1]:6.0f} m³/d   what happened: {flow(d['y_test'][i])[-1]:6.0f} m³/d")
print("                    (no storm in the training data: the storm is where every model struggles)")
```
<p class="src">workshop/examples/lesson5_train_and_evaluate.py, lines 77–88</p>


- `torch.save` writes the weights together with the mean and standard deviation and the layer sizes. A model saved without its scaler cannot be used on new raw measurements.
- The reload builds the same network from the saved sizes and loads the weights.
- `z = (np.log(raw) - blob["mean"]) / blob["std"]` scales raw readings with the saved numbers, never with numbers computed from new data.

```text
1-2 TRAIN+VALIDATE  MLP: best epoch 38; train loss 0.021, validation loss 0.308
    OVERFITTING     415k knobs on 200 examples: train loss 0.349 → 0.0075, validation 0.251 at epoch 9 → 0.381 at the end  (it memorised)
3 TEST (once)       error 2 h ahead on storm 2: persistence 4392   linear 2702   MLP 2747  m³/d
4 SAVE + RELOAD     calm, end of test  forecast in 2 h:  18556 m³/d   what happened:  18409 m³/d
4 SAVE + RELOAD     during storm 2     forecast in 2 h:  42122 m³/d   what happened:  28843 m³/d
                    (no storm in the training data: the storm is where every model struggles)
```
<p class="src">Output of <code>python workshop/examples/lesson5_train_and_evaluate.py</code>, from the run that built this guide</p>


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

```python
d = pd.read_excel(DATA / "Inf_dry_2006.xlsx").astype(float).values[:-1]
load = d[:, 14] * (d[:, 10] + 0.2 * d[:, 2])                            # flow × (ammonium + some COD)
LOAD = np.repeat(load / load.mean(), 3)                                  # 15 min → 5 min steps
```
<p class="src">workshop/examples/lesson6_rl.py, lines 22–24</p>


`d[:, 14]` is the flow, `d[:, 10]` ammonium and `d[:, 2]` the readily biodegradable COD. Dividing by the mean makes the load a factor around 1. `np.repeat(..., 3)` turns 15-minute samples into 5-minute steps by holding each value three times.

```python
class OxygenTank:
    ACTIONS = np.array([-20.0, -5.0, 0.0, 5.0, 20.0])                    # change of the air (KLa)

    def __init__(self, day, energy_weight=0.1, seed=0):
        self.day, self.w, self.rng = day, energy_weight, np.random.default_rng(seed)

    def reset(self):
        self.k0, self.k, self.kla, self.S, self.e1 = int(self.day * 288), 0, 120.0, 2.0, 0.0
        return self.state()

    def state(self):
        return np.array([self.S - 2.0, self.e1, self.kla / 240.0, LOAD[self.k0 + min(self.k, 287)]])

    def step(self, dkla):
        self.e1 = self.S - 2.0
        self.kla = float(np.clip(self.kla + dkla, 0.0, 240.0))
        our = 700.0 * LOAD[self.k0 + self.k] * (1 + 0.05 * self.rng.normal())
        s_eq = 8.0 - our / max(self.kla, 1e-6)
        self.S = max(0.0, s_eq + (self.S - s_eq) * np.exp(-self.kla * 5 / 1440))
        self.k += 1
        # ── 2 REWARD: be at 2 mg/L, don't waste air
        reward = -abs(self.S - 2.0) - self.w * self.kla / 240.0
        return self.state(), reward, self.k >= 288
```
<p class="src">workshop/examples/lesson6_rl.py, lines 28–50</p>


- `ACTIONS` are the five moves allowed: lower the air by 20 or 5, hold, raise by 5 or 20.
- `reset()` starts a day: step 0, air at 120, oxygen at 2 mg/L.
- `state()` is what the agent sees: the oxygen error (S − 2), the previous error, the air setting as a fraction of 240, and the current load.
- `step()` applies the move, clipped to 0 to 240. `our` is the bacteria's demand with 5 % random noise. Then the oxygen is moved towards S_eq with the exact solution of the mass balance over 5 minutes: `S_eq + (S − S_eq) · e^(−KLa · Δt)`, where Δt = 5/1440 of a day. Using the exact solution keeps the simulation stable at any step size.
- The reward is minus the oxygen error, minus the cost of air. One day is 288 steps of 5 minutes.

```python
pid = lambda s: float(np.clip(80 * (-s[0] + s[1]) + 25 * -s[0], -20, 20))   # the expert: a PID, no learning
print(f"1 ENVIRONMENT  PID on an unseen day: error {run(OxygenTank(10), pid)[1]:.3f} mg/L")
```
<p class="src">workshop/examples/lesson6_rl.py, lines 60–61</p>


The expert, an incremental PID. The error is e = 2 − S, which is `-s[0]`, and the previous error is `-s[1]`. The change of air is 80 × (e − e_previous) + 25 × e. The first term reacts to how fast the error changes, the second pushes steadily until the error is gone. No data, no training.

```python
E, K, L = np.array([-.5, -.25, -.1, .1, .25, .5]), np.linspace(0, 1, 17)[1:-1], np.array([.6, .8, 1., 1.2, 1.4])
cell = lambda s: (np.digitize(s[0], E), np.digitize(s[2], K), np.digitize(s[3], L))
greedy = lambda Q: (lambda s: OxygenTank.ACTIONS[int(np.argmax(Q[cell(s)]))])
```
<p class="src">workshop/examples/lesson6_rl.py, lines 64–66</p>


Q-learning keeps a table. To use a table the state must be a cell, so `cell(s)` sorts the error into 7 bins, the air setting into 16 and the load into 6. `greedy(Q)` is the policy that picks the action with the highest value in the current cell.

```python
def q_learning(energy_weight=0.1, episodes=1500, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((7, 16, 6, 5))
    best = (-np.inf, None)
    for ep in range(episodes):
        eps = max(0.05, 1 - ep / (0.6 * episodes))                      # explore a lot at first, then less
        env = OxygenTank(int(rng.integers(0, 7)), energy_weight, seed=ep)
        s, done = env.reset(), False
        while not done:
            c = cell(s)
            a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[c]))   # ── 3 ACT
            s, r, done = env.step(OxygenTank.ACTIONS[a])
            target = r + (0.0 if done else 0.9 * Q[cell(s)].max())
            Q[c + (a,)] += 0.2 * (target - Q[c + (a,)])                  # ── 4 LEARN (Bellman update)
        if ep % 100 == 99:
            v = run(OxygenTank(6, energy_weight), greedy(Q))[0]
            if v > best[0]:
                best = (v, Q.copy())
    return best[1]
```
<p class="src">workshop/examples/lesson6_rl.py, lines 69–87</p>


- `Q = np.zeros((7, 16, 6, 5))`: one value for every cell and every action, all starting at zero.
- `eps` is the exploration rate. It starts at 1 (every move random) and falls to 0.05 over the first 60 % of training.
- Each episode is one random day from week 1.
- **Act.** With probability `eps` the agent tries a random move, otherwise the best one it knows.
- **Learn.** `target = r + 0.9 × best value of the next cell` is the Bellman target: the reward now plus the discounted value of where we landed. The table entry moves 20 % of the way towards it.
- Every 100 episodes the greedy policy is tested on day 6 and the best table is kept, the same idea as keeping the best epoch.

{{code:workshop/examples/lesson6_rl.py|ff = {"last": None}|print(f"    BASELINE}}

The fair baseline. The agent sees the load, so we give the PID the same information the engineer's way. From the mass balance, holding 2 mg/L needs KLa = OUR / (8 − 2) = 700 × load / 6. The feedforward adds the change of that value to the PID's move.

```text
1 ENVIRONMENT  PID on an unseen day: error 0.304 mg/L
3-4 Q-LEARNING trained on days 0-6; on unseen day 10: error 0.235 mg/L
    BASELINE   PID + physics feedforward, no training: error 0.230 mg/L
2 REWARD       air made 50× dearer → the agent sets the air to 0 (PID: about 120); oxygen error 1.53 mg/L
```
<p class="src">Output of <code>python workshop/examples/lesson6_rl.py</code>, from the run that built this guide</p>


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

```python
            # eq. (9): y = r + gamma Q'(s', mu'(s'))
            with torch.no_grad():
                y = r + c.gamma * (1.0 - d) * self.critic_target(
                    torch.cat([o2, self.actor_target(o2)], dim=-1))
            q = self.critic(torch.cat([o, a], dim=-1))
            loss_c = ((q - y) ** 2).mean()                      # eq. (8)
            self.opt_critic.zero_grad(set_to_none=True)
            loss_c.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
            self.opt_critic.step()

            # eq. (10): maximise Q(s, mu(s)), plus the pre-activation penalty
            pre = self.actor.pre_activation(o)
            loss_a = (-self.critic(torch.cat([o, torch.tanh(pre)], dim=-1)).mean()
                      + c.pre_act_penalty * (pre ** 2).mean())
            self.opt_actor.zero_grad(set_to_none=True)
            loss_a.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
            self.opt_actor.step()
```
<p class="src">src/wwtp/rl/ddpg.py, lines 222–240</p>


- `y = r + γ · (1 − d) · Q_target(s′, μ_target(s′))` is the Bellman target, made with the slow target networks so it does not jump every step. `(1 − d)` drops the future at the end of an episode.
- `loss_c` is the squared error between the critic's estimate and that target. This is the same regression as in part 1.
- `loss_a` is minus the critic's value of the actor's action. Minimising it means the actor climbs the critic. The small penalty on `pre` keeps the `tanh` output from getting stuck at a limit.

### Slide 52 · Start from the expert, then improve, behind a safety gate

The project's controller in four boxes. The expert PID is run on dry week 1 and its moves recorded. A network copies it (behaviour cloning, which is supervised learning). Soft actor-critic improves on that copy using the reward. A safety gate switches the learned part off whenever the inputs leave the range seen in training. The code on the left is the copying step, a plain regression.

### Slide 53 · Papers 1 and 3 exactly as printed, and why they don't learn

Two code tours. Paper 1's MAACC is a small adaptive-critic controller added on top of a PID.

```python
    def step(self, z: np.ndarray) -> float:
        """One period of Algorithm 1 for this agent; returns ``a_k``."""
        zi = z[self.i]
        delta = bipolar(self.eta * zi)
        da = float(self.alpha @ delta)                   # eq. (14)
        a = self.a + da                                  # eq. (8)
        x = np.array([zi, da, a])
        vt = bipolar(self.theta @ x)
        q = float(self.beta @ vt)                        # eq. (13)
        u_k = self.utility(z, da, a)
        if self.prev is not None:                        # eqs. (15), (16), (18)
            vt_prev, q_prev = self.prev
            e = u_k + LAMBDA * q - q_prev
            if self.critic == "paper":
                g = e * LAMBDA * vt
            elif self.critic == "td0":
                g = -e * vt_prev
            else:                                        # residual gradient
                g = e * (LAMBDA * vt - vt_prev)
            self.beta = self.beta - LC * g
            q = float(self.beta @ vt)
        # eqs. (17), (19), (27): dL_a/dalpha = Q * (beta^T C) * delta
        dq_da = float(self.beta @ (0.5 * (1.0 - vt ** 2) * self.theta[:, 2]))
        self.alpha = self.alpha - LA * q * dq_da * delta
        self.prev = (vt, q)
        self.a = a
        return a
```
<p class="src">src/wwtp/maacc/exact.py, lines 67–93</p>


- The action network `alpha @ bipolar(eta * zi)` gives a change of input, which is added up into `a`. Because `alpha` starts at zero, the controller starts exactly as its PID.
- The critic estimates the value `q` and the utility `U` scores the moment, with the weights printed in the paper's Table I.
- The update of `alpha` is the learning rate times `q` times the slope of the critic. With the printed weights that product is about 10⁻⁴, so `alpha` hardly moves. In 80 of 80 runs, MAACC equals its PID to within 1 %. The code is correct. The method, as printed, cannot learn on this plant.

Paper 3's IOQL repeats 50 critic updates and 60 actor updates on each single sample. That many steps on one point makes it unstable: it either stays at its PID or the oxygen collapses.

### Slide 54 · ILC, the control step, the safety gate and the training script

Our controller in four files. The first is the daily ILC, iterative learning control. Each night it looks at what the PID had to do during the day, and the next day it does part of that in advance.

```python
    def _learn(self):
        new = self.f.copy()
        kern = np.ones(2 * self.w + 1) / (2 * self.w + 1)
        for i in range(len(self.beta)):
            ok = self._ok[:, i] if self.antiwindup else np.ones(PERIOD, bool)
            if ok.mean() < 0.5:                  # mostly at a limit today: keep yesterday's profile
                continue
            dev = np.where(ok, self._u[:, i] - self._u[ok, i].mean(), 0.0)
            step = self.beta[i] * np.roll(dev, -self.lead[i])
            step[np.roll(~ok, -self.lead[i])] = 0.0
            upd = self.f[:, i] + step
            pad = np.concatenate([upd[PERIOD - self.w:], upd, upd[: self.w]])     # the day is periodic
            new[:, i] = self.q * np.convolve(pad, kern, mode="valid")
        self.f = new
```
<p class="src">src/wwtp/sacil/ilc.py, lines 72–85</p>


- `_learn` runs once a day, at midnight.
- `ok` marks the moments when the blower was free. When the blower is flat out or off, what the PID asked for is not what the plant got, so those moments teach nothing. If more than half the day was spent at a limit, the ILC keeps yesterday's profile (`continue`). This is the anti-windup, added after held-out test 1.
- `dev` is the PID's deviation from its daily mean: the part of its work that repeats every day.
- `step` takes a share `beta` of that and moves it a few steps earlier (`np.roll`), so the air arrives just ahead of the load peak instead of just after it.
- The moving average smooths the profile over the day, treating midnight as joined to the next morning.

The control step adds the other half of the anti-windup.

```python
        aw = self.ilc is not None and self.ilc.antiwindup
        ff = 0.0 if self.ilc is None else self.ilc.ff(self.k)
        # anti-windup with ILC: the PID's own limits move with the feedforward, so that
        # PID + feedforward can reach either actuator limit and the PID never winds up
        lo, hi = (U_MIN - ff, U_MAX - ff) if aw else (U_MIN, U_MAX)
        if du is not None:
            self.u = np.clip(self.u + du, lo, hi)
        elif aw:
            self.u = np.clip(self.u, lo, hi)
        u_apply = self.u if self.ilc is None else np.clip(self.u + ff, U_MIN, U_MAX)
        sat = (self.u <= lo + 1e-9) | (self.u >= hi - 1e-9) if aw else None
```
<p class="src">src/wwtp/sacil/sacil.py, lines 155–165</p>


The PID's own limits are shifted by the feedforward `ff`, so PID plus feedforward together can reach either end of the blower's range and no more. Without this the PID keeps adding correction while the blower is already at its limit (it winds up), and then needs a long time to unwind.

The third tab is the safety gate.

```python
    def act(o, e):
        stats["n"] += 1
        if gated:
            ood = np.any((o < lo - MARGIN * width) | (o > hi + MARGIN * width))
            need = abs(expert_increment(e.e, e.e1, e.e2, demo_gap(e))[0]) > RES_SCALE[0]
            if ood or need:
                stats["gated"] += 1
                e.fallback_gain = GAIN if e.terms is None else 1.0
                return np.zeros(2)
            o = np.clip(o, lo, hi)
        return net(np.asarray(o, np.float32))
```
<p class="src">experiments/33_paper1_study.py, lines 145–155</p>


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

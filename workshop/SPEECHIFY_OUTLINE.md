# Workshop deck for an AI presentation builder (Speechify or similar)

Paste-ready text for building the workshop deck in an AI presentation tool. Each slide has a **title**, **bullets** for the slide, the **image** to upload (still pictures of the animated slides, in `workshop/slides/stills/`), and a **narration** the tool can read as the voice-over or keep as speaker notes.

How to use it:
1. Paste the whole "Deck" section below as the source text, or one slide at a time.
2. Upload the matching `slideNN.jpg` on each slide. They are 1920 × 1080 screenshots of `workshop/slides/index.html`.
3. Keep the live HTML deck for the interactive moments (the training fit, the layer-by-layer forward pass, gradient descent, the parameter counter). An AI-generated deck only shows them as pictures.

Tone: plain English, for research scholars new to ML. Total narration ≈ 35 minutes; the rest of the 90 minutes is the hands-on notebooks.

---

## Deck

### Slide 1 · From plant data to a trained model
- Section: Hands-on workshop · about 90 minutes
- Machine learning and reinforcement learning, explained with one real research project: learning to control a wastewater treatment plant (the IWA BSM1 benchmark).
- Image: `stills/slide01.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Welcome. Today we learn machine learning the way it is actually done in research: on one real project, from raw data to a trained model. The project is about a wastewater treatment plant. Bacteria clean the water, and they need oxygen, which we supply by blowing air. More air means cleaner water but more energy. Learning to control that trade-off is what our models try to do.

### Slide 2 · First the basics, then our project
- Section: Plan
- Part 1 explains how any machine-learning project goes, one step per slide, on a simple example. Parts 2 and 3 do the same steps on our research project, with its real files and code.
- Image: `stills/slide02.jpg`

**Narration:** The session has three parts. Part 1 is the basics: how any machine-learning project goes, one step per slide, on a simple example with no plant in it. Part 2 does the same steps on our research project, with its real files and code. Part 3 is reinforcement learning and the project's controller. Each part ends with a few questions and a notebook you run yourself.

### Slide 3 · Every word you will hear today, explained simply
- Section: Before we start · words in plain language
- Click any word. Each one has a plain meaning, an everyday comparison, and where it shows up in our project. Come back to this slide whenever a word is new.
- Image: `stills/slide03.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before we start, let's agree on the words. Every term you'll hear today is on this slide, with a plain meaning and an everyday comparison. A model is a recipe whose amounts you can adjust. Training is tasting and adjusting. The loss is the points you lose for a wrong guess. Overfitting is learning last year's exam answers by heart. Reinforcement learning is training a dog with treats. If you get lost later, come back to this page.

### Slide 4 · The basics
- Section: Part 1 of 3 · 40 min
- In this part: How any machine-learning project goes, one step per slide: from a raw file to a tested, saved model. The example is deliberately simple: predict tomorrow's water demand of a town from the weather.
- Hands-on: Notebook 0: 00_ml_skeleton.ipynb , every step on the slides, runs in about 10 s. Terminal: python workshop/examples/lesson1_knobs.py (a model with two knobs)
- Image: `stills/slide04.jpg`

**Narration:** Part 1 of 3: The basics. How any machine-learning project goes, one step per slide: from a raw file to a tested, saved model. The example is deliberately simple: predict tomorrow's water demand of a town from the weather.

### Slide 5 · Every machine-learning project follows the same steps
- Section: The basics · the whole recipe
- We take them one per slide. The example: a town wants to know tomorrow's water demand , and it has three years of daily records of temperature, rainfall and demand.
- Image: `stills/slide05.jpg`

**Narration:** Every machine-learning project follows the same twelve steps. Collect the data, clean it, decide what goes in and what comes out, split it into training, validation and test, scale it, and feed it in. Then choose a model, train it, check its size against the data, choose its settings, test it once, and save it. Our example for part 1: a town wants to predict tomorrow's water demand from the weather.

### Slide 6 · Where data comes from, and the first look
- Section: The basics · step 1 · collect
- Files from instruments or colleagues: CSV, Excel, text logs.
- Databases and exports from software.
- Public datasets and web APIs.
- Simulations , when measuring is too slow or too dangerous.
- Image: `stills/slide06.jpg`

**Narration:** Data comes from files, databases, public datasets or simulations. Whatever the source, write down where it came from, which version, the units of every column, and whether you may share it. Reading it is one line of pandas. Then look before doing anything else: the shape, the type of each column, the first rows, and the minimum and maximum of every column. In our example that first look already shows a demand ten times too big and a negative rainfall.

### Slide 7 · Cleaning: find what is wrong, fix only what you can explain
- Section: The basics · step 2 · clean
- Image: `stills/slide07.jpg`

**Narration:** Cleaning means finding what is wrong and fixing only what you can explain. Wrong types are converted, duplicate rows dropped, short gaps filled from their neighbours, long gaps left out, impossible values set to unknown, and spikes found by comparing each value with the median of its neighbours. Keep a list of every fix, so someone else can rebuild your dataset.

### Slide 8 · Decide what goes in (X) and what comes out (y)
- Section: The basics · step 3 · inputs and target
- One row = one example. The model learns the link from the blue columns to the orange one.
- Only put in X what you will really have when you make the prediction. Tomorrow's real temperature is not known today; tomorrow's forecast is.
- The shape is the first thing to know. A table: (examples, inputs). A time series: (examples, time steps, channels). Images: (examples, height, width, colours).
- Image: `stills/slide08.jpg`

**Narration:** One row is one example. The inputs, X, are what we know today: tomorrow's weather forecast, whether tomorrow is a weekend, and today's demand. The target, y, is tomorrow's demand. Only put in X what you will really have when you make the prediction. And know the shape: here 1094 examples of 4 inputs and 1 target.

### Slide 9 · Train, validation and test: three separate piles
- Section: The basics · step 4 · split
- Image: `stills/slide09.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** We split the examples into three piles. The model learns from the training pile. We make our choices with the validation pile. The test pile gives the final score, and we look at it once. When the data comes in time order, split by time: a random split puts near-copies of training examples into the test, and the score looks far better than it is.

### Slide 10 · Put every input on a similar scale
- Section: The basics · step 5 · scale
- log for quantities that only grow by factors (flows, loads): storms stop dominating.
- standardise : z = (x − mean) / std, per channel.
- Compute mean and std on the training data only , then reuse those numbers for validation, test and later use. Otherwise the test set leaks into training.
- Image: `stills/slide10.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Inputs on very different scales confuse a network: a flow of twenty thousand would drown a temperature of 25. So we take the log of quantities that grow by factors, then standardise: subtract the mean, divide by the standard deviation. The mean and standard deviation come from the training data only, and the same numbers are used for everything after, so the scaler is saved with the model.

### Slide 11 · How the data gets into the model: tensors and batches
- Section: The basics · step 6 · feed
- Tensor : an array that PyTorch can do maths on, and follow back for the gradients.
- Batch : a few examples at a time. Faster than one by one, and steadier.
- Epoch : one pass through all the batches. Training runs for many epochs.
- Shuffle the training batches; never the validation or test data.
- Image: `stills/slide11.jpg`

**Narration:** The data reaches the model as tensors, arrays PyTorch can compute with and trace back for gradients. A data loader shuffles the training examples and serves them in batches of 32. The model turns each batch of 32 by 4 inputs into 32 guesses at once. One pass through all the batches is an epoch.

### Slide 12 · What is a model?
- Section: The basics · step 7 · the model
- Loss : one number for "how wrong": the mean squared length of the red lines.
- Training : turn the knobs a little in the direction that lowers the loss (the gradient). Repeat.
- A neural network is the same thing with thousands of knobs and a more flexible function.
- Image: `stills/slide12.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** A model is a function with knobs. Here it is a straight line, y hat equals w times x plus b. The input x is what we know, y hat is the model's guess, and the two knobs w and b are its parameters. The red lines are its mistakes; their average square is the loss. Training turns each knob a little in the direction that lowers the loss, again and again. A neural network is the same thing with thousands of knobs.

### Slide 13 · Three kinds of learning: it depends on what answers you have
- Section: The basics · step 7 · kinds of learning
- Regression : y is a number. Tomorrow's demand, a temperature, a price.
- Classification : y is a class. Spam or not, which fault, which species.
- Clustering : groups of similar days or samples.
- Dimension reduction (PCA): many columns summarised by a few.
- Anomaly detection : what does not look like the rest.
- An agent tries actions, sees what happens, and learns which actions pay off.
- Image: `stills/slide13.jpg`

**Narration:** There are three kinds of learning. Supervised: you have the right answer for every example, a number for regression or a class for classification. Unsupervised: there is no answer, and the model finds structure, such as groups of similar samples. Reinforcement: no right answers at all, only a reward after acting. Our demand example is supervised regression.

### Slide 14 · The shape of your data picks the kind of model
- Section: The basics · step 7 · kinds of models
- Image: `stills/slide14.jpg`

**Narration:** The shape of the data picks the kind of model. A table: start with a linear model, trees, or a small network. A sequence: a convolution over time, a recurrent network or a Transformer. An image: a convolutional network. The output picks the last layer and the loss. And always start simple: a baseline with no learning, then a linear model, and only then something bigger.

### Slide 15 · What happens inside a neural network, layer by layer
- Section: The basics · step 7 · inside a network
- Each neuron: weighted sum + bias , then an activation (here ReLU: negatives become 0).
- A whole layer at once: h = ReLU(W·x + b) . That's one matrix multiplication.
- Line colour is the weight's sign (blue +, red −); thickness is its size. Training only changes these numbers.
- Image: `stills/slide15.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** What happens inside a neural network, layer by layer. Each neuron: weighted sum + bias, then an activation (here ReLU: negatives become 0). A whole layer at once: h = ReLU(W·x + b). That's one matrix multiplication. Line colour is the weight's sign (blue +, red −); thickness is its size. Training only changes these numbers.

### Slide 16 · The loss says what "wrong" means
- Section: The basics · step 8 · train (1)
- MSE (e²): punishes big errors hard; gives the mean.
- MAE (|e|): robust to outliers; gives the median.
- Pinball at τ: under-predicting costs τ, over-predicting 1−τ. At τ = 0.9 the best answer is the value exceeded only 10 % of the time. Three τ values give a band, for example a 10/50/90 % range.
- Cross-entropy for classes; binary cross-entropy for "violation: yes or no".
- Image: `stills/slide16.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The loss is your definition of wrong, and choosing it is a design decision. Squared error punishes big mistakes. Absolute error is robust to outliers. The pinball loss lets a model predict a range, which tells an operator how sure the forecast is.

### Slide 17 · Gradient descent: walk downhill on the loss
- Section: The basics · step 8 · train (2)
- Contours are the loss over two weights; the ball is the model.
- Too small : it barely moves. Too large : it overshoots and blows up. Try all three.
- Backpropagation computes ∂loss/∂w for every weight in one backward pass: loss.backward() .
- In practice: Adam (it adapts the step for every knob), sometimes a learning-rate schedule and gradient clipping .
- Image: `stills/slide17.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Gradient descent: walk downhill on the loss. Contours are the loss over two weights; the ball is the model. Too small: it barely moves. Too large: it overshoots and blows up. Try all three. Backpropagation computes ∂loss/∂w for every weight in one backward pass: loss.backward().

### Slide 18 · What training code looks like
- Section: The basics · step 8 · train (3)
- Image: `stills/slide18.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is what training code looks like. For each batch: run the model, compute the loss, clear the old gradients, compute new ones with backward, and let the optimiser turn the knobs. After every epoch, measure the loss on the validation data with learning switched off, and keep the knobs from the best epoch. The curves show why: training loss keeps falling, but validation loss turns up when the model starts to memorise.

### Slide 19 · Counting parameters, and comparing with your data
- Section: The basics · step 9 · how big?
- Image: `stills/slide19.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** How big should the model be? Count its knobs. A layer from n inputs to m outputs has n times m weights plus m biases. Our small model has 97 knobs for 765 training examples, so it cannot memorise much. A model with a million knobs for the same data could learn every example by heart. The shapes fix the first and last layers; the amount of data limits what goes in between.

### Slide 20 · Parameters are learned; settings are chosen by you
- Section: The basics · step 10 · settings
- The recipe: try a few settings, keep the one with the lowest validation loss, then test that one, once.
- Good starting points: a small model, Adam with learning rate 10⁻³ to 10⁻², stop at the best validation epoch.
- Grow the model only while the validation loss keeps falling.
- Image: `stills/slide20.jpg`

**Narration:** Parameters are learned by training. Settings, also called hyperparameters, are chosen by you: the number of units, the learning rate, the batch size, the number of epochs. Try a few and keep the one with the lowest validation loss. Here the middle size wins: 4 units are too few, 64 start to memorise. Bigger is not automatically better.

### Slide 21 · What testing code looks like
- Section: The basics · step 11 · test
- Same preparation as training: the same columns, the same scaler.
- Real units. "44 thousand litres a day" means something; "0.22" does not.
- Against a baseline. The simplest guess you can think of. If the model can't beat it, it is not worth having.
- Once. If you go back and change the model after looking at the test score, the test has become validation.
- Numbers: MAE, RMSE, R². Classes: accuracy, precision and recall, the confusion matrix.
- Image: `stills/slide21.jpg`

**Narration:** Testing comes last, after the settings are chosen. Put back the best knobs, switch learning off, predict the test examples, and turn the predictions back into real units. Then compare with a baseline, the simplest guess there is: tomorrow equals today. Our model is off by 44 thousand litres a day, the baseline by 57. Test once: if you change the model after looking, the test has become validation.

### Slide 22 · One notebook, twelve steps: the skeleton of every ML notebook
- Section: The basics · the whole skeleton
- Image: `stills/slide22.jpg`

**Narration:** Here is the whole of part 1 on one slide: the skeleton of every machine-learning notebook. Imports, collect, clean, inputs and target, split, scale, feed, model, train, size, settings, test, save. Every notebook you will meet, including our project's, is these blocks in this order. When a notebook confuses you, ask which of the twelve steps a cell belongs to.

### Slide 23 · Notebook 0: the skeleton
- Section: Your turn · 10 minutes
- Remove demand_today from the inputs. How much worse is the test error?
- Set hidden to 256. More parameters: better or worse on validation?
- Try a learning rate of 1.0. What happens to the loss?
- Replace the split with a random one. Does the test score look better? Should you believe it?
- Image: `stills/slide23.jpg`

**Narration:** Your turn. Open notebook 0 and run it from top to bottom; it takes about ten seconds. Then change one thing at a time: remove today's demand from the inputs, make the model much bigger, use a huge learning rate, or split at random, and see what each change does to the validation and test errors.

### Slide 24 · Four questions before we move on
- Section: Part 1 · check yourself
- Q: Which parts of ŷ = w·x + b does training change? → A: Only w and b, the knobs (parameters). The inputs x and the true answers y are data and never change. The shape of the function, and settings such as the learning rate, are chosen by us before training.
- Q: Where do the mean and std for scaling come from? → A: From the training data only. The same numbers are then used for validation, test and every later prediction, so they are saved with the model.
- Q: Training loss falls, validation loss rises. What is happening? → A: Overfitting: the model is memorising the training examples. Keep the best validation epoch, use a smaller model or regularisation, or get more data.
- Q: Why look at the test set only once? → A: Every decision made after looking at it turns it into validation data. Then it no longer tells you how the model does on new data.
- Image: `stills/slide24.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Part 1 ends with four questions. Ask the room first, then reveal each answer. Which parts of ŷ = w·x + b does training change? Only w and b, the knobs (parameters). The inputs x and the true answers y are data and never change. The shape of the function, and settings such as the learning rate, are chosen by us before training. Where do the mean and std for scaling come from? From the training data only. The same numbers are then used for validation, test and every later prediction, so they are saved with the model. Training loss falls, validation loss rises. What is happening? Overfitting: the model is memorising the training examples. Keep the best validation epoch, use a smaller model or regularisation, or get more data. Why look at the test set only once? Every decision made after looking at it turns it into validation data. Then it no longer tells you how the model does on new data.

### Slide 25 · Our project: plant data to a forecaster
- Section: Part 2 of 3 · 30 min
- In this part: The same steps on our research project: the BSM1 plant, its official Excel files, our cleaning (with a physics fix), cutting a time series into examples, choosing a model for it, training and testing it.
- Hands-on: Notebook 1 (collect, clean, prepare) and notebook 2 (choose and train) Terminal: lesson2_load_and_clean.py to lesson5_train_and_evaluate.py
- Image: `stills/slide25.jpg`

**Narration:** Part 2 of 3: Our project: plant data to a forecaster. The same steps on our research project: the BSM1 plant, its official Excel files, our cleaning (with a physics fix), cutting a time series into examples, choosing a model for it, training and testing it.

### Slide 26 · What the project is about
- Section: Our project · the problem
- The plant: BSM1, the standard test plant in this field. Five tanks and a settler. The bacteria remove pollution; they need oxygen.
- What we measure: oxygen in tank 5 (S O,5 ), nitrate in tank 2 (S NO,2 ), and what flows in: flow Q, ammonium (S NH ) and COD (the organic dirt). The inflow is sampled every 15 min; the controller acts every 45 s.
- What we can change: the air blown into tank 5 (KLa 5 : how fast oxygen dissolves, 0–240 per day; more air = higher KLa) and the recycle pump (Q a ).
- The goal: hold oxygen at 2 mg/L and nitrate at 1 mg/L whatever arrives, without wasting energy.
- What we did: rebuilt three published learning controllers, checked their claims, and trained forecasters, a digital twin, a warning model and RL agents.
- Image: `stills/slide26.jpg`

**Narration:** The project has four machine-learning tasks, and each is a different kind of learning. Forecasting the inflow is regression. Warning before ammonium breaks the legal limit is classification. Predicting how the plant responds is a sequence model. Deciding how much air to blow is reinforcement learning. Today we follow the forecasting task from start to finish, then see how the others differ.

### Slide 27 · The plant and its two control loops
- Section: Our project · the plant
- Image: `stills/slide27.jpg`

**Narration:** This is the plant as the project drew it. Water flows from left to right. Tanks 1 and 2 get no air, so bacteria there remove nitrate. Tanks 3 to 5 are aerated, so other bacteria turn ammonium into nitrate, and the internal recycle carries that nitrate back to tank 1. The settler lets the sludge sink, and most of it goes back to the start. Our controllers move two things: the air in tank 5, to hold 2 milligrams of oxygen per litre, and the recycle pump, to hold 1 milligram of nitrate per litre in tank 2.

### Slide 28 · Our data: the official BSM1 influent files
- Section: Our project · step 1 · collect
- The IWA files: 14 days × 96 samples/day = 1345 rows , 15 columns (time, 13 concentrations, flow).
- Three weathers: dry, rain, storm. Week 1 is the same dry week in all three: a trap for splitting, as we'll see.
- Image: `stills/slide28.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Now our own data. The official BSM1 influent files: what flows into the plant, every 15 minutes for 14 days, which is 1345 rows of 15 columns: the time, 13 concentrations and the flow. There are three files, for dry, rain and storm weather. The first week is the same dry week in all three, which matters when we split.

### Slide 29 · Load it once, record where it came from, cache it
- Section: Our project · step 1 · reading the Excel files
- Pin the version. A URL with a commit hash always returns the same bytes. "The data on the website" changes.
- Keep only real data rows. Spreadsheets carry titles, units and notes. The filter keeps rows whose first cell is a number.
- Cache in a fast format. Parsing xlsx is slow; .npy loads in milliseconds.
- Don't copy data you may not redistribute. These files are GPL; the repository downloads them at run time.
- Image: `stills/slide29.jpg`

**Narration:** Four habits for collecting data. Pin the exact version you used, so anyone can rebuild your dataset. Keep only the real data rows. Save a fast copy so you don't parse the spreadsheet every time. And respect the licence: these files are downloaded at run time, not copied into the repository.

### Slide 30 · Cleaning our files, and a fix only physics can find
- Section: Our project · step 2 · clean
- Text rows (titles, units): drop them, but write the units down.
- Missing values : count them first. Short gaps → interpolate in time; long gaps → drop the window.
- Duplicate timestamps : keep one. Then check the step is a constant 15 min.
- Spikes (×10 here: a unit or typing error): flag against a rolling median, then fix or drop.
- Physics check. S_I stays at 30 while rain doubles the flow, so the rain should have diluted it. The official rain file has S_I = 30·Q dry /Q , and the project applies the same rule ( plant.py:89 ).
- Missing columns the model needs: TSS = 0.75·(X I +X S +X BH +X BA +X P ) ( plant.py:90 ).
- Image: `stills/slide30.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Cleaning our files. The usual problems from part 1 come back: text rows with titles and units, missing values, duplicate times and spikes. But the official files had one problem no statistical check finds. The inert COD, S_I, stays at 30 even when rain doubles the flow. Rain water is clean, so it must dilute the sewage. The project applies the dilution rule, S_I equals 30 times the dry flow divided by the flow, and writes the fix down.

### Slide 31 · Put everything on one time grid
- Section: Our project · step 2 · one time grid
- Sensors sample every 15 min, but the controller acts every 45 s. Between samples the influent is held constant (a zero-order hold): 20 control steps per sample.
- Image: `stills/slide31.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Different instruments work at different rates. The influent is sampled every fifteen minutes but the controller acts every forty-five seconds, so each sample is held for twenty control steps. Always put all your signals on one time grid before you build a dataset.

### Slide 32 · Download, read, check, fix
- Section: Our project · where is the code?
- The same steps (download, read, check and fix), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson2_load_and_clean.py
- Image: `stills/slide32.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step, download, read, check and fix, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 33 · Collect and clean, cell by cell
- Section: Our project · notebook 1 running
- Image: `stills/slide33.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before you start, here is the notebook running. Each cell has its real output, and the panel explains the idea behind it.

### Slide 34 · Turn one long series into many (input, target) examples
- Section: Our project · step 3 · inputs and target
- Image: `stills/slide34.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Our data is one long time series, not a table, so we cut it into examples. A window slides along the series: the last 24 hours are the input, the next 2 hours are the target. The input always ends where the target begins, so the model only sees the past. Stacked up, the inputs have the shape N by 96 time steps by 3 channels, and the targets N by 8. We split by weather and time: dry and rain to train, the first storm to validate, the second storm to test.

### Slide 35 · Choose, split, scale, cut: cell by cell
- Section: Our project · notebook 1 running
- Image: `stills/slide35.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Notebook 1, part C, running cell by cell: choosing the inputs, splitting by time, scaling with the training numbers, cutting the windows and checking their shapes.

### Slide 36 · Notebook 1: collect, clean, prepare
- Section: Your turn · 15 minutes
- Download the three official files (cached after the first time)
- Open one raw with pandas: what is in the first rows?
- Name the columns, check the shape, the types and the time step
- Plot the three weathers. Is week 1 really the same in all three?
- Run the checks: NaN, duplicates, negative values, ranges
- Find and fix the problems in a damaged copy
- Image: `stills/slide36.jpg`

**Narration:** Your turn: notebook 1, parts A, B and C. Download and open the official files, check and clean them, apply the dilution fix, then choose the inputs, split, scale and cut the windows. If the notebook gives trouble, the lesson 2 and lesson 3 scripts do the same in the terminal.

### Slide 37 · Our shapes, and the models in the project
- Section: Our project · step 7 · which model?
- Image: `stills/slide37.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Which model for our data? The table from part 1 answers it. The input is a sequence, 96 time steps of 3 measurements, and the output is 8 numbers. A sequence in and numbers out: a one-dimensional convolution over time, with a linear last layer. But first the baselines, then a linear model, then an MLP. On the right are the other models in the project: different shapes, so different models.

### Slide 38 · A causal convolution: the same small filter slides along time
- Section: Our project · step 7 · the forecaster's building block
- Each output looks at 3 inputs (kernel 3), only from the past: padding is added on the left ( F.pad(x, (pad, 0)) , forecast/models.py:38 ).
- Dilation 1, 2, 4… makes each layer look twice as far back, so a few layers cover a whole day.
- Image: `stills/slide38.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** A causal convolution: the same small filter slides along time. Each output looks at 3 inputs (kernel 3), only from the past: padding is added on the left (F.pad(x, (pad, 0)), forecast/models.py:38). Dilation 1, 2, 4… makes each layer look twice as far back, so a few layers cover a whole day.

### Slide 39 · Shapes, baselines and models, cell by cell
- Section: Our project · notebook 2 running
- Image: `stills/slide39.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Here the concepts become code. The layer-by-layer cell prints the shape after every layer of the real network, and the causal network proves in two lines that it cannot see the future.

### Slide 40 · Training, overfitting and the test, cell by cell
- Section: Our project · notebook 2 running
- Image: `stills/slide40.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The training loop in real code, then overfitting on purpose: four hundred thousand parameters on two hundred examples. Finally the test, used once, where the simplest model wins.

### Slide 41 · Train, validate, test once, save
- Section: Our project · where is the code?
- The same steps (train, validate, test once and save), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson5_train_and_evaluate.py
- Image: `stills/slide41.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step, train, validate, test once and save, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 42 · A model is only good compared with something simple
- Section: Our project · step 9 · test against baselines
- Persistence : "the next 2 h equal now". Free, and hard to beat at short horizons.
- Seasonal naive : "the same time yesterday". Strong when there is a daily cycle.
- Linear model, gradient-boosted trees. If a network can't beat these, it isn't earning its keep ( warning/baselines.py ).
- Report errors in physical units (m³/d, mg/L), per horizon, on the test set, over several seeds.
- Image: `stills/slide42.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Always compare with something simple. If a free rule like "the next two hours equal now" is as good as your network, the network isn't earning its keep. This project learned it the hard way: a well-tuned classical controller matched the learned one. That is not a failure, it is a result, and it belongs in the report.

### Slide 43 · Notebook 2: choose and train
- Section: Your turn · 20 minutes
- Baselines first: persistence and same time yesterday
- A linear model (scikit-learn): count its parameters
- An MLP in PyTorch: print every layer's output shape for one batch
- The training loop: train vs validation curves, keep the best epoch
- A small causal CNN (the forecaster's building block)
- Overfit on purpose, then fix it
- Image: `stills/slide43.jpg`

**Narration:** Your turn: notebook 2. Build the baselines, a linear model, a small network and a causal CNN, train them with a validation set, overfit on purpose, then test once and save the model with its scaler. Fill in the table of parameters and test errors as you go.

### Slide 44 · Four questions before we move on
- Section: Part 2 · check yourself
- Q: Why did every column of the Excel file come out as text? → A: The first row holds names. One text row makes pandas read the whole column as text. Tell it which row is the header, and keep only the numeric rows.
- Q: All the usual checks passed. Can the data still be wrong? → A: Yes: S_I stays at 30 while rain doubles the flow. Only physics (rain dilutes) catches it.
- Q: Read the shape (1906, 96, 3). → A: 1 906 examples, each with 96 time steps, each step with 3 measurements (flow, ammonium, COD).
- Q: Why did the simplest model win on the storm? → A: About 1 900 examples and no storm in training: the flexible models had little to learn from and more room to overfit. More varied data helps more than a bigger model.
- Image: `stills/slide44.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Part 2 ends with four questions. Ask the room first, then reveal each answer. Why did every column of the Excel file come out as text? The first row holds names. One text row makes pandas read the whole column as text. Tell it which row is the header, and keep only the numeric rows. All the usual checks passed. Can the data still be wrong? Yes: S_I stays at 30 while rain doubles the flow. Only physics (rain dilutes) catches it. Read the shape (1906, 96, 3). 1 906 examples, each with 96 time steps, each step with 3 measurements (flow, ammonium, COD). Why did the simplest model win on the storm? About 1 900 examples and no storm in training: the flexible models had little to learn from and more room to overfit. More varied data helps more than a bigger model.

### Slide 45 · Reinforcement learning and our controller
- Section: Part 3 of 3 · 20 min
- In this part: Learning from a reward instead of right answers. Then the project itself: how it was built, the published methods we rebuilt, our controller and the latest results.
- Hands-on: Notebook 3: reinforcement learning in brief Terminal: python workshop/examples/lesson6_rl.py
- Image: `stills/slide45.jpg`

**Narration:** Part 3 of 3: Reinforcement learning and our controller. Learning from a reward instead of right answers. Then the project itself: how it was built, the published methods we rebuilt, our controller and the latest results.

### Slide 46 · When there is no correct answer to copy
- Section: Reinforcement learning · the idea
- Image: `stills/slide46.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Part 3 starts with the idea of reinforcement learning. When we control the air, there is no right answer to copy. So an agent tries actions, the plant responds, and a reward scores what happened. The reward is where you write down what you want. Paper 2's reward pays for air and for ammonium above the limit; its agent saved energy and broke the ammonium limit far more often. It followed the reward exactly as written.

### Slide 47 · The parts of an actor-critic agent (DDPG, SAC)
- Section: Reinforcement learning · the parts
- Image: `stills/slide47.jpg`

**Narration:** The parts of an actor-critic agent: the environment, an actor network that picks the action, a critic network that estimates the future reward, a replay buffer of past experience, slow target copies of the networks, and noise for exploring. The critic is trained exactly like the regression in part 1, against a target it makes for itself.

### Slide 48 · Reinforcement learning, cell by cell
- Section: Reinforcement learning · notebook 3 running
- Image: `stills/slide48.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The RL notebook running, cell by cell, with its real results.

### Slide 49 · Notebook 3: reinforcement learning in brief
- Section: Your turn · 12 minutes
- A one-tank aeration environment with reset() and step() , driven by the real daily load
- The PID baseline
- Imitation: copy the PID with a small network (supervised learning)
- Q-learning from reward alone: watch the learning curve
- Compare all three on a day never used for training
- Image: `stills/slide49.jpg`

**Narration:** Your turn: notebook 3. A one-tank aeration environment driven by the real daily load, a PID as the expert, a network that copies it, and a Q-learning agent that learns from the reward alone. Then change the energy weight in the reward and watch what the agent does with the oxygen.

### Slide 50 · How the project was built: one script per step
- Section: Our project · end to end
- Image: `stills/slide50.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before the details, here is the whole project on one page. Each box is one script. We first rebuilt the official plant and data, then reproduced the three published controllers exactly as printed, then built our own controller and two strong baselines, tested everything, and wrote the report from the result files. Click any box to see the command and what it found.

### Slide 51 · Reward, DDPG update and SAC-IL update, in the project's code
- Section: Our project · RL code
- Image: `stills/slide51.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Now the reinforcement-learning code of the project. The reward is two lines, and what it leaves out explains what the agent did. The critic update is just a regression. Our SAC-IL keeps imitating the expert while it learns.

### Slide 52 · Start from the expert, then improve, behind a safety gate
- Section: Our project · our learned controller
- Learning from scratch on a plant means weeks of bad actions. Starting from the expert removes that.
- Result: −74 to −87 % oxygen error against the paper's PID. But a PID tuned term by term with the ILC did as well, and on new sensors with noise every high-gain controller failed. Held-out test 2, three slides on, shows what fixed part of that and what is still open.
- Image: `stills/slide52.jpg`

**Narration:** Learning from scratch on a real plant means weeks of bad decisions. So the project's controller first copies the existing expert controller, which is ordinary supervised learning, and only then improves with reward. A safety gate hands control back to the expert whenever conditions look unfamiliar.

### Slide 53 · Papers 1 and 3 exactly as printed, and why they don't learn
- Section: Our project · reproducing the papers
- Image: `stills/slide53.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is reproduction in practice: the published equations, line by line, in code. Paper 1's learning signal is so small that the controller never leaves its PID. Paper 3 repeats so many updates on one sample that it becomes unstable.

### Slide 54 · ILC, the control step, the safety gate and the training script
- Section: Our project · our controller
- Image: `stills/slide54.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Our controller, in four files. The daily learning control, the control step that combines PID, feedforward and the learned correction, the safety gate that hands control back to the expert, and the training script from start to end.

### Slide 55 · The second held-out test: two fixes, one open problem
- Section: Our project · the latest results
- Fix 1, ILC anti-windup: time spent at a blower limit teaches the ILC nothing, and the PID's limits move with the feedforward.
- Fix 2, robust PID: a 0.75-minute filter on the oxygen reading and half the gains.
- Protocol pushed before any run; new influent windows and new noise seeds ( docs/PAPER1_HELDOUT2.md ).
- Image: `stills/slide55.jpg`

**Narration:** These are the project's latest results. The first held-out test found two flaws, so two fixes were made: anti-windup for the daily learning control, and a robust PID with a filter on the oxygen reading and half the gains. They were tested on new influent and new sensor noise, with the plan committed before any run. Anti-windup removed the learning control's harm under overload and cost nothing in normal weather. The robust PID never failed and was 83 percent better than the paper's PID on new influent. But with the official sensor noise, nothing we built beats the paper's own gentle PID. That is the open problem, and it is reported.

### Slide 56 · Four questions before we move on
- Section: Part 3 · check yourself
- Q: What is the difference between a reward and a label? → A: A label is the right answer for this example. A reward only scores what happened, often late, and never says what the right action was.
- Q: Air made very expensive in the reward: what did the agent do? → A: Switched the air almost off and let the oxygen fall. It did exactly what the reward said, not what we meant.
- Q: What does the safety gate do? → A: When an input leaves the range seen in training, the learned part is switched off and the expert PID acts alone.
- Q: What did held-out test 2 show? → A: Anti-windup removed the ILC's harm under overload, and the robust PID never failed. But under the official sensor noise nothing beat the paper's own PID. That is the open problem.
- Image: `stills/slide56.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Part 3 ends with four questions. Ask the room first, then reveal each answer. What is the difference between a reward and a label? A label is the right answer for this example. A reward only scores what happened, often late, and never says what the right action was. Air made very expensive in the reward: what did the agent do? Switched the air almost off and let the oxygen fall. It did exactly what the reward said, not what we meant. What does the safety gate do? When an input leaves the range seen in training, the learned part is switched off and the expert PID acts alone. What did held-out test 2 show? Anti-windup removed the ILC's harm under overload, and the robust PID never failed. But under the official sensor noise nothing beat the paper's own PID. That is the open problem.

### Slide 57 · Questions learners ask, with short answers
- Section: Questions people ask
- Q: Do I need a GPU? → A: No. Every script and notebook here runs on a laptop CPU in seconds to a minute. GPUs matter for big models on big data, not for this.
- Q: Where does the gradient formula 2·(pred − y)·x in lesson1_knobs.py come from? → A: It is the slope of the loss (pred − y)² when you nudge w: calculus (the chain rule). In real code loss.backward() computes it for every knob automatically.
- Q: How big should my model be? → A: Start small and simple (a baseline, then a linear model). Grow only while the validation error keeps improving. Far more knobs than examples means it can memorise.
- Q: My data has no time order. Do I still split by time? → A: No: then a random split is fine. Split by whatever must be new at test time: time, plant, patient, sensor.
- Q: Why not pick the best model on the test set? → A: Then the test score is part of your choices and no longer tells you about new data. Choose on validation, test once.
- Q: Can the RL agent run on a real plant? → A: Not straight away: early exploration means bad actions. The project starts from the expert (imitation), keeps a safety gate, and tests in simulation first.
- Q: Did anything fix the sensor-noise problem? → A: Not yet. A filter and lower gains stop the failures, but with 0.25 g/m³ of noise on a 2 mg/L set point the paper's gentle PID is still as good. Next ideas: a Kalman filter, or switching gains when noise is high.
- Q: What should I put in my paper? → A: The data source and version, every correction, the split, the baselines, several seeds, and the test result, whichever way it went.
- Image: `stills/slide57.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Questions learners ask, with short answers. Do I need a GPU? No. Every script and notebook here runs on a laptop CPU in seconds to a minute. GPUs matter for big models on big data, not for this. Where does the gradient formula 2·(pred − y)·x in lesson1_knobs.py come from? It is the slope of the loss (pred − y)² when you nudge w: calculus (the chain rule). In real code loss.backward() computes it for every knob automatically. How big should my model be? Start small and simple (a baseline, then a linear model). Grow only while the validation error keeps improving. Far more knobs than examples means it can memorise. My data has no time order. Do I still split by time? No: then a random split is fine. Split by whatever must be new at test time: time, plant, patient, sensor. Why not pick the best model on the test set? Then the test score is part of your choices and no longer tells you about new data. Choose on validation, test once. Can the RL agent run on a real plant? Not straight away: early exploration means bad actions. The project starts from the expert (imitation), keeps a safety gate, and tests in simulation first. Did anything fix the sensor-noise problem? Not yet. A filter and lower gains stop the failures, but with 0.25 g/m³ of noise on a 2 mg/L set point the paper's gentle PID is still as good. Next ideas: a Kalman filter, or switching gains when noise is high. What should I put in my paper? The data source and version, every correction, the split, the baselines, several seeds, and the test result, whichever way it went.

### Slide 58 · What this project teaches about doing ML research
- Section: Take home
- Look at the data first. Units, gaps, physics. The S_I column needed a fix nobody had written down.
- The input and output shapes choose the model family ; the amount of data limits its size.
- Split by time or by plant , fit the scaler on training data only, never tune on the test.
- Baselines first : persistence, a linear model, a well-tuned PID.
- Reproduce before you improve. Two of three published methods did not work as printed.
- Several seeds. DDPG's energy saving ranged from 5.6 to 15.7 % between seeds.
- Image: `stills/slide58.jpg`

**Narration:** To finish, eight habits from this project. Look at your data. Let the shapes choose the model. Split honestly. Beat simple baselines. Reproduce published work before claiming to improve it. Use several seeds. Write your test down before you run it. And save your model with everything it needs to be used again. Everything we showed is in the repository. Questions?


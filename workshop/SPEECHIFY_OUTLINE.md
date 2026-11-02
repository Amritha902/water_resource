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

### Slide 2 · One lesson, seven steps, 90 minutes
- Section: Plan
- Every step has the same rhythm: the idea (animated) → where the code is → the code running → three questions → your turn. Every step points to the file in the repository that does it for real.
- Image: `stills/slide02.jpg`

**Narration:** One lesson, seven steps, 90 minutes. Plan One lesson, seven steps, 90 minutes Every step has the same rhythm: the idea (animated) → where the code is → the code running → three questions → your turn. Every step points to the file in the repository that does it for real. 1What is a model? · 8 min · A model is a function with knobs. Trai

### Slide 3 · Every word you will hear today, explained simply
- Section: Before we start · words in plain language
- Click any word. Each one has a plain meaning, an everyday comparison, and where it shows up in our project. Come back to this slide whenever a word is new.
- Image: `stills/slide03.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before we start, let's agree on the words. Every term you'll hear today is on this slide, with a plain meaning and an everyday comparison. A model is a recipe whose amounts you can adjust. Training is tasting and adjusting. The loss is the points you lose for a wrong guess. Overfitting is learning last year's exam answers by heart. Reinforcement learning is training a dog with treats. If you get lost later, come back to this page.

### Slide 4 · What the project is about
- Section: The problem
- The plant: BSM1, the standard test plant in this field. Five tanks and a settler. The bacteria remove pollution; they need oxygen.
- What we measure: oxygen in tank 5 (S O,5 ), nitrate in tank 2 (S NO,2 ), and what flows in: flow Q, ammonium (S NH ) and COD (the organic dirt). The inflow is sampled every 15 min; the controller acts every 45 s.
- What we can change: the air blown into tank 5 (KLa 5 : how fast oxygen dissolves, 0–240 per day; more air = higher KLa) and the recycle pump (Q a ).
- The goal: hold oxygen at 2 mg/L and nitrate at 1 mg/L whatever arrives, without wasting energy.
- What we did: rebuilt three published learning controllers, checked their claims, and trained forecasters, a digital twin, a warning model and RL agents.
- Image: `stills/slide04.jpg`

**Narration:** The project has four machine-learning tasks, and each is a different kind of learning. Forecasting the inflow is regression. Warning before ammonium breaks the legal limit is classification. Predicting how the plant responds is a sequence model. Deciding how much air to blow is reinforcement learning. Today we follow the forecasting task from start to finish, then see how the others differ.

### Slide 5 · How the project was built: one script per step
- Section: Our code, end to end
- Image: `stills/slide05.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before the details, here is the whole project on one page. Each box is one script. We first rebuilt the official plant and data, then reproduced the three published controllers exactly as printed, then built our own controller and two strong baselines, tested everything, and wrote the report from the result files. Click any box to see the command and what it found.

### Slide 6 · What is a model?
- Section: Lesson 1 of 7 · 8 min
- Goal: A model is a function with knobs. Training turns the knobs, in four steps that every model uses.
- Terminal: python workshop/examples/lesson1_knobs.py
- Notebook: none needed; slide 8 shows the same lines in notebook 2 and the project.
- Image: `stills/slide06.jpg`

**Narration:** Lesson 1 of 7: What is a model? After this lesson you will be able to say: A model is a function with knobs. Training turns the knobs, in four steps that every model uses.

### Slide 7 · What is a model?
- Section: Lesson 1 · the core idea
- Loss : one number for "how wrong": the mean squared length of the red lines.
- Training : turn the knobs a little in the direction that lowers the loss (the gradient). Repeat.
- A neural network is the same thing with thousands of knobs and a more flexible function.
- Image: `stills/slide07.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Everything today builds on this one picture. A model is a function with knobs. Here the function is a straight line with two knobs, its slope and its offset. The red lines show how wrong it is, and their average squared length is the loss. Training means turning the knobs a little in the direction that makes the loss smaller, again and again. A neural network is exactly this, only with thousands of knobs.

### Slide 8 · Four steps train every model
- Section: Lesson 1 · where is the code?
- The same steps (the four training steps), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson1_knobs.py
- Image: `stills/slide08.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, the four training steps, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 9 · Three questions before we move on
- Section: Lesson 1 · check yourself
- Q: Which parts of ŷ = w·x + b does training change? → A: Only w and b, the knobs. The inputs x and the true answers y are data and never change. The shape of the function is chosen by us before training.
- Q: The loss is very large. What does that tell you? → A: The guesses are far from the true answers: the knobs are not yet good, or the model's shape cannot fit the data.
- Q: What happens if the step size (learning rate) is far too big? → A: Each step jumps over the bottom of the valley and lands higher on the other side, further each time: the loss explodes. Try lr = 1.2 in lesson1_knobs.py and see it.
- Image: `stills/slide09.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 1 ends with three questions. Ask the room first, then reveal each answer. Which parts of ŷ = w·x + b does training change? Only w and b, the knobs. The inputs x and the true answers y are data and never change. The shape of the function is chosen by us before training. The loss is very large. What does that tell you? The guesses are far from the true answers: the knobs are not yet good, or the model's shape cannot fit the data. What happens if the step size (learning rate) is far too big? Each step jumps over the bottom of the valley and lands higher on the other side, further each time: the loss explodes. Try lr = 1.2 in lesson1_knobs.py and see it.

### Slide 10 · Collect and clean the data
- Section: Lesson 2 of 7 · 20 min
- Goal: Download data reproducibly, read it without traps, check it, and fix what is wrong, including the physics.
- Terminal: python workshop/examples/lesson2_load_and_clean.py
- Notebook: notebook 1, parts A and B
- Image: `stills/slide10.jpg`

**Narration:** Lesson 2 of 7: Collect and clean the data After this lesson you will be able to say: Download data reproducibly, read it without traps, check it, and fix what is wrong, including the physics.

### Slide 11 · Where the data comes from
- Section: Lesson 2 · collect
- The IWA files: 14 days × 96 samples/day = 1345 rows , 15 columns (time, 13 concentrations, flow).
- Three weathers: dry, rain, storm. Week 1 is the same dry week in all three: a trap for splitting, as we'll see.
- Image: `stills/slide11.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Our data is what flows into the plant: the flow rate and thirteen concentrations, every fifteen minutes, for two weeks. There are three weather scenarios. Notice that the first week is identical in all three files. If we are careless when we split the data, we will test on what we trained on. Keep that in mind.

### Slide 12 · Load it once, record where it came from, cache it
- Section: Lesson 2 · collect · the code
- Pin the version. A URL with a commit hash always returns the same bytes. "The data on the website" changes.
- Keep only real data rows. Spreadsheets carry titles, units and notes. The filter keeps rows whose first cell is a number.
- Cache in a fast format. Parsing xlsx is slow; .npy loads in milliseconds.
- Don't copy data you may not redistribute. These files are GPL; the repository downloads them at run time.
- Image: `stills/slide12.jpg`

**Narration:** Four habits for collecting data. Pin the exact version you used, so anyone can rebuild your dataset. Keep only the real data rows. Save a fast copy so you don't parse the spreadsheet every time. And respect the licence: these files are downloaded at run time, not copied into the repository.

### Slide 13 · Cleaning: look before you learn
- Section: Lesson 2 · clean
- Text rows (titles, units): drop them, but write the units down.
- Missing values : count them first. Short gaps → interpolate in time; long gaps → drop the window.
- Duplicate timestamps : keep one. Then check the step is a constant 15 min.
- Spikes (×10 here: a unit or typing error): flag against a rolling median, then fix or drop.
- Physics check. S_I stays at 30 while rain doubles the flow, so the rain should have diluted it. The official rain file has S_I = 30·Q dry /Q , and the project applies the same rule ( plant.py:89 ).
- Missing columns the model needs: TSS = 0.75·(X I +X S +X BH +X BA +X P ) ( plant.py:90 ).
- Image: `stills/slide13.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before learning anything, look at the data. Real sensor data has missing values, duplicates, spikes and stuck sensors. Statistics finds most of these. But one error in this project was only found with physics: one concentration stayed constant while rain doubled the flow, which is impossible, because rain dilutes it. The project corrects it and documents the correction. Every correction you make should be written down.

### Slide 14 · Put everything on one time grid
- Section: Lesson 2 · clean · time grid
- Sensors sample every 15 min, but the controller acts every 45 s. Between samples the influent is held constant (a zero-order hold): 20 control steps per sample.
- Image: `stills/slide14.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Different instruments work at different rates. The influent is sampled every fifteen minutes but the controller acts every forty-five seconds, so each sample is held for twenty control steps. Always put all your signals on one time grid before you build a dataset.

### Slide 15 · Download, read, check, fix
- Section: Lesson 2 · where is the code?
- The same steps (download, read, check and fix), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson2_load_and_clean.py
- Image: `stills/slide15.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, download, read, check and fix, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 16 · How the project loads and corrects the official data
- Section: Lesson 2 · our code
- Image: `stills/slide16.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is the project's own code for what we just discussed. Step through it: the raw table comes from the cache, the columns are laid out the way the plant simulator expects, and one line applies the rain-dilution fix.

### Slide 17 · Collect and clean, cell by cell
- Section: Lesson 2 · the notebook running
- Image: `stills/slide17.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before you start, here is the notebook running. Each cell has its real output, and the panel explains the idea behind it.

### Slide 18 · Three questions before we move on
- Section: Lesson 2 · check yourself
- Q: Why did every column come out as text? → A: The first row holds names. One text row makes pandas read the whole column as text. Tell it row 0 is the header.
- Q: All checks passed. Can the data still be wrong? → A: Yes: S_I stays at 30 while rain doubles the flow. Only physics (rain dilutes) catches it.
- Q: When may you fill a gap by interpolation? → A: Only when it is short (here ≤ 1 hour). Filling a long gap invents data.
- Image: `stills/slide18.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 2 ends with three questions. Ask the room first, then reveal each answer. Why did every column come out as text? The first row holds names. One text row makes pandas read the whole column as text. Tell it row 0 is the header. All checks passed. Can the data still be wrong? Yes: S_I stays at 30 while rain doubles the flow. Only physics (rain dilutes) catches it. When may you fill a gap by interpolation? Only when it is short (here ≤ 1 hour). Filling a long gap invents data.

### Slide 19 · Notebook 1, parts A and B
- Section: Your turn · 12 minutes
- Download the three official files (cached after the first time)
- Open one raw with pandas: what is in the first rows?
- Name the columns, check the shape, the types and the time step
- Plot the three weathers. Is week 1 really the same in all three?
- Run the checks: NaN, duplicates, negative values, ranges
- Find and fix the problems in a damaged copy
- Image: `stills/slide19.jpg`

**Narration:** Notebook 1, parts A and B. Download the three official files (cached after the first time) Open one raw with pandas: what is in the first rows? Name the columns, check the shape, the types and the time step

### Slide 20 · Prepare the data
- Section: Lesson 3 of 7 · 12 min
- Goal: Choose inputs and target, split by time, scale with training numbers only, cut examples.
- Terminal: python workshop/examples/lesson3_prepare.py
- Notebook: notebook 1, part C
- Image: `stills/slide20.jpg`

**Narration:** Lesson 3 of 7: Prepare the data After this lesson you will be able to say: Choose inputs and target, split by time, scale with training numbers only, cut examples.

### Slide 21 · Transform and scale the inputs
- Section: Lesson 3 · prepare
- log for quantities that only grow by factors (flows, loads): storms stop dominating.
- standardise : z = (x − mean) / std, per channel.
- Compute mean and std on the training data only , then reuse those numbers for validation, test and deployment. Otherwise the test set leaks into training.
- Image: `stills/slide21.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Networks learn best when every input sits around zero with a similar spread. Flow is around twenty thousand, ammonium around thirty; without scaling, flow would dominate. We take the log, then standardise. One rule matters most: compute the mean and standard deviation on the training data only. Otherwise information from the test set leaks into training.

### Slide 22 · Turn one long series into many (input, target) examples
- Section: Lesson 3 · prepare
- Image: `stills/slide22.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Turn one long series into many (input, target) examples. Lesson 3 · prepare Turn one long series into many (input, target) examples # src/wwtp/forecast/dataset.py:67 LOOKBACK = 96 (24 h), HORIZON = 8 (2 h) def _windows(obs, stride): n = obs.shape[0] - LOOKBACK - HORIZON starts = np.arange(0, n, stride) xs = np.stack([obs[s:s+LOOKBACK] for s in starts]) #

### Slide 23 · Split by time, not at random
- Section: Lesson 3 · prepare
- Neighbouring windows share 95 % of their samples. Split them at random and the test set holds near-copies of training examples, so the score looks far better than reality.
- Train on the past, validate on later data, test on data that nothing was tuned on.
- In this project: paper 2: learn on week 1, test on week 2 ( rl/train.py )
- warning model: separate simulated plants for train, val and test (seeds 0, 1000, 2000 in experiments/04 )
- paper 1: a held-out test whose protocol was committed before it was run ( docs/PAPER1_HELDOUT_PROTOCOL.md )
- Image: `stills/slide23.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** With time series, never split at random. Neighbouring windows overlap almost completely, so a random split hides copies of the training data in the test set, and the score looks far better than reality. Split by time instead. The project went further: it wrote down its final test before running it, and reported the result whichever way it went.

### Slide 24 · Choose, split, scale, cut
- Section: Lesson 3 · where is the code?
- The same steps (choose, split, scale and cut), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson3_prepare.py
- Image: `stills/slide24.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, choose, split, scale and cut, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 25 · Prepare, cell by cell
- Section: Lesson 3 · the notebook running
- Image: `stills/slide25.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The preparation steps in code. Watch the shapes: ninety-six steps, three channels, eight targets.

### Slide 26 · Three questions before we move on
- Section: Lesson 3 · check yourself
- Q: Why not split the windows at random? → A: Neighbouring windows share 95 of 96 samples: the test would hold near-copies of training examples and the score would be far too good.
- Q: Where do the mean and std for scaling come from? → A: From the training data only. Then the same numbers are used for validation, test and the real plant.
- Q: Read the shape (1906, 96, 3). → A: 1 906 examples, each with 96 time steps, each step with 3 measurements (flow, ammonium, COD).
- Image: `stills/slide26.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 3 ends with three questions. Ask the room first, then reveal each answer. Why not split the windows at random? Neighbouring windows share 95 of 96 samples: the test would hold near-copies of training examples and the score would be far too good. Where do the mean and std for scaling come from? From the training data only. Then the same numbers are used for validation, test and the real plant. Read the shape (1906, 96, 3). 1 906 examples, each with 96 time steps, each step with 3 measurements (flow, ammonium, COD).

### Slide 27 · Notebook 1, part C: prepare
- Section: Your turn · 10 minutes
- Pick the inputs a real plant can measure at the inlet: flow, ammonium, COD (as forecast/dataset.py )
- Split: train = dry (14 d) + rain week 2; validation = storm days 7–10.25 (storm 1); test = storm days 10.25–14 (storm 2)
- log, then standardise with training statistics
- Cut windows: 96 past steps → 8 future steps of flow
- Check the shapes and that no window crosses a split
- Save everything to prepared.npz
- Image: `stills/slide27.jpg`

**Narration:** Notebook 1, part C: prepare. Pick the inputs a real plant can measure at the inlet: flow, ammonium, COD (as forecast/dataset.py) Split: train = dry (14 d) + rain week 2; validation = storm days 7–10.25 (storm 1); test = storm days 10.25–14 (storm 2) log, then standardise with training statistics

### Slide 28 · Choose the model
- Section: Lesson 4 of 7 · 10 min
- Goal: Let the input and output shapes pick the model family; look inside a network layer by layer; count its knobs.
- Terminal: python workshop/examples/lesson4_models.py
- Notebook: notebook 2, sections 1–6
- Image: `stills/slide28.jpg`

**Narration:** Lesson 4 of 7: Choose the model After this lesson you will be able to say: Let the input and output shapes pick the model family; look inside a network layer by layer; count its knobs.

### Slide 29 · The input and output decide the model
- Section: Lesson 4 · choose a model
- Image: `stills/slide29.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** How do you choose a model? Start from the shapes. If your input is a sequence, use a model that knows about time order. If your output is a number, use a linear last layer and a squared or absolute error. If it is a probability, use a sigmoid and cross-entropy. The six models of this project each follow these rules. The amount of data then limits how big the model can be.

### Slide 30 · What happens inside a neural network, layer by layer
- Section: Lesson 4 · inside the model
- Each neuron: weighted sum + bias , then an activation (here ReLU: negatives become 0).
- A whole layer at once: h = ReLU(W·x + b) . That's one matrix multiplication.
- Line colour is the weight's sign (blue +, red −); thickness is its size. Training only changes these numbers.
- Image: `stills/slide30.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** What happens inside a neural network, layer by layer. Each neuron: weighted sum + bias, then an activation (here ReLU: negatives become 0). A whole layer at once: h = ReLU(W·x + b). That's one matrix multiplication. Line colour is the weight's sign (blue +, red −); thickness is its size. Training only changes these numbers.

### Slide 31 · Counting parameters
- Section: Lesson 4 · size of the model
- Image: `stills/slide31.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Count your parameters and compare them with your data. Our small network has about twenty-three thousand parameters for nineteen hundred examples. The reinforcement-learning agent of one paper has sixty-seven thousand. When parameters far outnumber examples, the model can simply memorise the training set, and we will see that happen in the notebook.

### Slide 32 · A causal convolution: the same small filter slides along time
- Section: Lesson 4 · models for sequences (1)
- Each output looks at 3 inputs (kernel 3), only from the past: padding is added on the left ( F.pad(x, (pad, 0)) , forecast/models.py:38 ).
- Dilation 1, 2, 4… makes each layer look twice as far back, so a few layers cover a whole day.
- Image: `stills/slide32.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** A causal convolution: the same small filter slides along time. Each output looks at 3 inputs (kernel 3), only from the past: padding is added on the left (F.pad(x, (pad, 0)), forecast/models.py:38). Dilation 1, 2, 4… makes each layer look twice as far back, so a few layers cover a whole day.

### Slide 33 · A recurrent network carries a memory through time
- Section: Lesson 4 · models for sequences (2)
- At every step the same cell reads the new input and the previous memory h, and writes a new h.
- The digital twin ( twin/models.py ): input [oxygen, nitrate, air, recycle, inflow…], output the next oxygen and nitrate.
- Image: `stills/slide33.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** A recurrent network carries a memory through time. At every step the same cell reads the new input and the previous memory h, and writes a new h. The digital twin (twin/models.py): input [oxygen, nitrate, air, recycle, inflow…], output the next oxygen and nitrate.

### Slide 34 · Shapes → baselines → linear → MLP → CNN
- Section: Lesson 4 · where is the code?
- The same steps (shapes, baselines, linear, MLP and CNN), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson4_models.py
- Image: `stills/slide34.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, shapes, baselines, linear, MLP and CNN, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 35 · Shapes, baselines and models, cell by cell
- Section: Lesson 4 · the notebook running
- Image: `stills/slide35.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Here the concepts become code. The layer-by-layer cell prints the shape after every layer of the real network, and the causal network proves in two lines that it cannot see the future.

### Slide 36 · Three questions before we move on
- Section: Lesson 4 · check yourself
- Q: How many knobs has a layer from 64 to 8 units? → A: 64 × 8 weights + 8 biases = 520.
- Q: What does ReLU do? → A: Keeps positive numbers, turns negative ones into 0. That bend is what lets a network learn curved patterns.
- Q: Why is the convolution 'causal'? → A: It is padded on the left only, so each output sees only the present and the past, never the future.
- Image: `stills/slide36.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 4 ends with three questions. Ask the room first, then reveal each answer. How many knobs has a layer from 64 to 8 units? 64 × 8 weights + 8 biases = 520. What does ReLU do? Keeps positive numbers, turns negative ones into 0. That bend is what lets a network learn curved patterns. Why is the convolution 'causal'? It is padded on the left only, so each output sees only the present and the past, never the future.

### Slide 37 · Train and evaluate
- Section: Lesson 5 of 7 · 20 min
- Goal: Train with a validation set, spot overfitting, beat the baselines, test once, save the model with its scaler.
- Terminal: python workshop/examples/lesson5_train_and_evaluate.py
- Notebook: notebook 2, sections 5–9
- Image: `stills/slide37.jpg`

**Narration:** Lesson 5 of 7: Train and evaluate After this lesson you will be able to say: Train with a validation set, spot overfitting, beat the baselines, test once, save the model with its scaler.

### Slide 38 · The loss says what "wrong" means
- Section: Lesson 5 · train
- MSE (e²): punishes big errors hard; gives the mean.
- MAE (|e|): robust to outliers; gives the median.
- Pinball at τ: under-predicting costs τ, over-predicting 1−τ. At τ = 0.9 the best answer is the value exceeded only 10 % of the time. Three τ values give a band: the forecaster's 10/50/90 % range.
- Cross-entropy for classes; binary cross-entropy for "violation: yes or no".
- Image: `stills/slide38.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The loss is your definition of wrong, and choosing it is a design decision. Squared error punishes big mistakes. Absolute error is robust to outliers. The pinball loss lets a model predict a range, which tells an operator how sure the forecast is.

### Slide 39 · Gradient descent: walk downhill on the loss
- Section: Lesson 5 · train
- Contours are the loss over two weights; the ball is the model.
- Too small : it barely moves. Too large : it overshoots and blows up. Try all three.
- Backpropagation computes ∂loss/∂w for every weight in one backward pass: loss.backward() .
- In practice: Adam/AdamW (adaptive steps), a learning-rate schedule (OneCycle, cosine) and gradient clipping , as in every training file of the repo.
- Image: `stills/slide39.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Gradient descent: walk downhill on the loss. Contours are the loss over two weights; the ball is the model. Too small: it barely moves. Too large: it overshoots and blows up. Try all three. Backpropagation computes ∂loss/∂w for every weight in one backward pass: loss.backward().

### Slide 40 · The training loop
- Section: Lesson 5 · train
- Image: `stills/slide40.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Every training script in the world is this loop. Take a batch, predict, compute the loss, compute the gradients, update the weights. After each pass through the data, check the validation loss and keep the best model. When training loss keeps falling but validation loss rises, the model has started to memorise.

### Slide 41 · Train, validate, test once, save
- Section: Lesson 5 · where is the code?
- The same steps (train, validate, test once and save), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson5_train_and_evaluate.py
- Image: `stills/slide41.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, train, validate, test once and save, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 42 · Training, overfitting and the test, cell by cell
- Section: Lesson 5 · the notebook running
- Image: `stills/slide42.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The training loop in real code, then overfitting on purpose: four hundred thousand parameters on two hundred examples. Finally the test, used once, where the simplest model wins.

### Slide 43 · A model is only good compared with something simple
- Section: Lesson 5 · evaluate
- Persistence : "the next 2 h equal now". Free, and hard to beat at short horizons.
- Seasonal naive : "the same time yesterday". Strong when there is a daily cycle.
- Linear model, gradient-boosted trees. If a network can't beat these, it isn't earning its keep ( warning/baselines.py ).
- Report errors in physical units (m³/d, mg/L), per horizon, on the test set, over several seeds.
- Image: `stills/slide43.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Always compare with something simple. If a free rule like "the next two hours equal now" is as good as your network, the network isn't earning its keep. This project learned it the hard way: a well-tuned classical controller matched the learned one. That is not a failure, it is a result, and it belongs in the report.

### Slide 44 · Three questions before we move on
- Section: Lesson 5 · check yourself
- Q: Training loss falls, validation loss rises. What is happening? → A: Overfitting: the model memorises the training examples. Keep the best epoch, regularise, use a smaller model or get more data.
- Q: Why use the test set only once? → A: Every look that changes a decision turns the test into training data. Then it no longer measures the future.
- Q: What must be saved with the weights? → A: The scaler (mean, std) and the settings. Without them nobody can feed new raw measurements to the model.
- Image: `stills/slide44.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 5 ends with three questions. Ask the room first, then reveal each answer. Training loss falls, validation loss rises. What is happening? Overfitting: the model memorises the training examples. Keep the best epoch, regularise, use a smaller model or get more data. Why use the test set only once? Every look that changes a decision turns the test into training data. Then it no longer measures the future. What must be saved with the weights? The scaler (mean, std) and the settings. Without them nobody can feed new raw measurements to the model.

### Slide 45 · Notebook 2: choose and train
- Section: Your turn · 20 minutes
- Baselines first: persistence and same time yesterday
- A linear model (scikit-learn): count its parameters
- An MLP in PyTorch: print every layer's output shape for one batch
- The training loop: train vs validation curves, keep the best epoch
- A small causal CNN (the forecaster's building block)
- Overfit on purpose, then fix it
- Image: `stills/slide45.jpg`

**Narration:** Notebook 2: choose and train. Baselines first: persistence and same time yesterday A linear model (scikit-learn): count its parameters An MLP in PyTorch: print every layer's output shape for one batch

### Slide 46 · Reinforcement learning
- Section: Lesson 6 of 7 · 13 min
- Goal: Learn from a reward instead of right answers: environment, reward, exploring, learning. Then the project's agents.
- Terminal: python workshop/examples/lesson6_rl.py
- Notebook: notebook 3
- Image: `stills/slide46.jpg`

**Narration:** Lesson 6 of 7: Reinforcement learning After this lesson you will be able to say: Learn from a reward instead of right answers: environment, reward, exploring, learning. Then the project's agents.

### Slide 47 · When there is no correct answer to copy
- Section: Lesson 6 · reinforcement learning
- Image: `stills/slide47.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** When there is no correct answer to copy. Lesson 6 · reinforcement learning When there is no correct answer to copy SupervisedReinforcement dataa fixed datasetcollected by acting signalthe right answer ya reward, often late learnsx → ystate → action (policy) riskwrong predictionswrong actions on a plant # src/wwtp/rl/rewards.py:39 paper 2,

### Slide 48 · The parts of an actor-critic agent (DDPG, SAC)
- Section: Lesson 6 · reinforcement learning
- Image: `stills/slide48.jpg`

**Narration:** The parts of an actor-critic agent (DDPG, SAC). Lesson 6 · reinforcement learning The parts of an actor-critic agent (DDPG, SAC) Environmentreset(), step(action) → next state, reward, done (rl/env.py) Actor (policy)network: state → action. DDPG: 3 → 256 → 256 → 1, tanh Criticnetwork: (state, action) → expected future reward Q Replay bufferstores

### Slide 49 · Environment, reward, act, learn
- Section: Lesson 6 · where is the code?
- The same steps (environment, reward, act and learn), each in its own colour, in four panels: the hands-on script, the notebook, and the project's own code
- Run it yourself: python workshop/examples/lesson6_rl.py
- Image: `stills/slide49.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Where is the code? These four panels are real files. Each step of this lesson, environment, reward, act and learn, has one colour, and it lights up in every panel at once: first in the short script you can run, then in the notebook, then in the project's own code. What you just learned is exactly what the research code does.

### Slide 50 · Reward, DDPG update and SAC-IL update, in the project's code
- Section: Lesson 6 · our code
- Image: `stills/slide50.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Now the reinforcement-learning code of the project. The reward is two lines, and what it leaves out explains what the agent did. The critic update is just a regression. Our SAC-IL keeps imitating the expert while it learns.

### Slide 51 · Reinforcement learning, cell by cell
- Section: Lesson 6 · the notebook running
- Image: `stills/slide51.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The RL notebook running, cell by cell, with its real results.

### Slide 52 · Start from the expert, then improve, behind a safety gate
- Section: Lesson 6 · the project's controller
- Learning from scratch on a plant means weeks of bad actions. Starting from the expert removes that.
- Result: −74 to −87 % oxygen error against the paper's PID. But a PID tuned term by term with the ILC did as well, and on new sensors with noise every high-gain controller failed. That honest result is in the README.
- Image: `stills/slide52.jpg`

**Narration:** Learning from scratch on a real plant means weeks of bad decisions. So the project's controller first copies the existing expert controller, which is ordinary supervised learning, and only then improves with reward. A safety gate hands control back to the expert whenever conditions look unfamiliar.

### Slide 53 · Three questions before we move on
- Section: Lesson 6 · check yourself
- Q: What is the difference between a reward and a label? → A: A label is the right answer for this example. A reward only scores what happened, often late, and never says what the right action was.
- Q: Why does the agent sometimes act at random? → A: To explore: without trying new actions it can never find out that one is better.
- Q: Air made very expensive in the reward: what did the agent do? → A: Switched the air almost off and let the oxygen fall. It did exactly what the reward said, not what we meant.
- Image: `stills/slide53.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 6 ends with three questions. Ask the room first, then reveal each answer. What is the difference between a reward and a label? A label is the right answer for this example. A reward only scores what happened, often late, and never says what the right action was. Why does the agent sometimes act at random? To explore: without trying new actions it can never find out that one is better. Air made very expensive in the reward: what did the agent do? Switched the air almost off and let the oxygen fall. It did exactly what the reward said, not what we meant.

### Slide 54 · Notebook 3: reinforcement learning in brief
- Section: Your turn · 12 minutes
- A one-tank aeration environment with reset() and step() , driven by the real daily load
- The PID baseline
- Imitation: copy the PID with a small network (supervised learning)
- Q-learning from reward alone: watch the learning curve
- Compare all three on a day never used for training
- Image: `stills/slide54.jpg`

**Narration:** Notebook 3: reinforcement learning in brief. A one-tank aeration environment with reset() and step(), driven by the real daily load The PID baseline Imitation: copy the PID with a small network (supervised learning)

### Slide 55 · Our project's code
- Section: Lesson 7 of 7 · 5 min
- Goal: Read the published methods and our controller in the project's own code, and the lessons they taught us.
- Terminal: none; this lesson reads the project's own files on the slides
- Notebook: none needed. To go deeper: docs/PAPER1_STUDY.md and README.md .
- Image: `stills/slide55.jpg`

**Narration:** Lesson 7 of 7: Our project's code After this lesson you will be able to say: Read the published methods and our controller in the project's own code, and the lessons they taught us.

### Slide 56 · Papers 1 and 3 exactly as printed, and why they don't learn
- Section: Lesson 7 · reproducing the papers
- Image: `stills/slide56.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is reproduction in practice: the published equations, line by line, in code. Paper 1's learning signal is so small that the controller never leaves its PID. Paper 3 repeats so many updates on one sample that it becomes unstable.

### Slide 57 · ILC, the control step, the safety gate and the training script
- Section: Lesson 7 · our controller
- Image: `stills/slide57.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Our controller, in four files. The daily learning control, the control step that combines PID, feedforward and the learned correction, the safety gate that hands control back to the expert, and the training script from start to end.

### Slide 58 · Three questions before we move on
- Section: Lesson 7 · check yourself
- Q: MAACC as printed equals its PID in 80 of 80 runs. Why? → A: Its learning signal with the printed weights is about 10⁻⁴: the networks hardly move from their zero start.
- Q: What does the safety gate do? → A: When an input leaves the range seen in training, the learned part is switched off and the expert PID acts alone.
- Q: Our learned controller vs a tuned PID + ILC: which won? → A: The tuned PID with the daily ILC matched or beat it on the test week. Reported honestly in the README.
- Image: `stills/slide58.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Lesson 7 ends with three questions. Ask the room first, then reveal each answer. MAACC as printed equals its PID in 80 of 80 runs. Why? Its learning signal with the printed weights is about 10⁻⁴: the networks hardly move from their zero start. What does the safety gate do? When an input leaves the range seen in training, the learned part is switched off and the expert PID acts alone. Our learned controller vs a tuned PID + ILC: which won? The tuned PID with the daily ILC matched or beat it on the test week. Reported honestly in the README.

### Slide 59 · Questions learners ask, with short answers
- Section: Lesson 7 · questions people ask
- Q: Do I need a GPU? → A: No. Every script and notebook here runs on a laptop CPU in seconds to a minute. GPUs matter for big models on big data, not for this.
- Q: Where does the gradient formula 2·(pred − y)·x in lesson 1 come from? → A: It is the slope of the loss (pred − y)² when you nudge w: calculus (the chain rule). In real code loss.backward() computes it for every knob automatically.
- Q: How big should my model be? → A: Start small and simple (a baseline, then a linear model). Grow only while the validation error keeps improving. Far more knobs than examples means it can memorise.
- Q: Why did the simplest model win on the storm? → A: About 1 900 examples and no storm in training: the flexible models had little to learn and more to overfit. More varied data helps more than a bigger model.
- Q: My data has no time order. Do I still split by time? → A: No: then a random split is fine. Split by whatever must be new at test time: time, plant, patient, sensor.
- Q: Why not pick the best model on the test set? → A: Then the test score is part of your choices and no longer tells you about new data. Choose on validation, test once.
- Q: Can the RL agent run on a real plant? → A: Not straight away: early exploration means bad actions. The project starts from the expert (imitation), keeps a safety gate, and tests in simulation first.
- Q: What should I put in my paper? → A: The data source and version, every correction, the split, the baselines, several seeds, and the test result, whichever way it went.
- Image: `stills/slide59.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Questions learners ask, with short answers. Do I need a GPU? No. Every script and notebook here runs on a laptop CPU in seconds to a minute. GPUs matter for big models on big data, not for this. Where does the gradient formula 2·(pred − y)·x in lesson 1 come from? It is the slope of the loss (pred − y)² when you nudge w: calculus (the chain rule). In real code loss.backward() computes it for every knob automatically. How big should my model be? Start small and simple (a baseline, then a linear model). Grow only while the validation error keeps improving. Far more knobs than examples means it can memorise. Why did the simplest model win on the storm? About 1 900 examples and no storm in training: the flexible models had little to learn and more to overfit. More varied data helps more than a bigger model. My data has no time order. Do I still split by time? No: then a random split is fine. Split by whatever must be new at test time: time, plant, patient, sensor. Why not pick the best model on the test set? Then the test score is part of your choices and no longer tells you about new data. Choose on validation, test once. Can the RL agent run on a real plant? Not straight away: early exploration means bad actions. The project starts from the expert (imitation), keeps a safety gate, and tests in simulation first. What should I put in my paper? The data source and version, every correction, the split, the baselines, several seeds, and the test result, whichever way it went.

### Slide 60 · What this project teaches about doing ML research
- Section: Lesson 7 · take home
- Look at the data first. Units, gaps, physics. The S_I column needed a fix nobody had written down.
- The input and output shapes choose the model family ; the amount of data limits its size.
- Split by time or by plant , fit the scaler on training data only, never tune on the test.
- Baselines first : persistence, a linear model, a well-tuned PID.
- Reproduce before you improve. Two of three published methods did not work as printed.
- Several seeds. DDPG's energy saving ranged from 5.6 to 15.7 % between seeds.
- Image: `stills/slide60.jpg`

**Narration:** To finish, eight habits from this project. Look at your data. Let the shapes choose the model. Split honestly. Beat simple baselines. Reproduce published work before claiming to improve it. Use several seeds. Write your test down before you run it. And save your model with everything it needs to be used again. Everything we showed is in the repository. Questions?


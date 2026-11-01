# Workshop deck for an AI presentation builder (Speechify or similar)

Paste-ready text for building the workshop deck in an AI presentation tool. Each slide has a **title**, **bullets** for the slide, the **image** to upload (still pictures of the animated slides, in `workshop/slides/stills/`), and a **narration** the tool can read as the voice-over or keep as speaker notes.

How to use it:
1. Paste the whole "Deck" section below as the source text, or one slide at a time.
2. Upload the matching `slideNN.jpg` on each slide. They are 1920 × 1080 screenshots of `workshop/slides/index.html`.
3. Keep the live HTML deck for the interactive moments (the training fit, the layer-by-layer forward pass, gradient descent, the parameter counter). An AI-generated deck only shows them as pictures.

Tone: plain English, for research scholars new to ML. Total narration ≈ 30 minutes; the rest of the 90 minutes is the hands-on notebooks.

---

## Deck

### Slide 1 · From plant data to a trained model
- Hands-on workshop on machine learning and reinforcement learning, about 90 minutes
- One real research project: learning to control a wastewater treatment plant (IWA BSM1 benchmark)
- Image: `stills/slide01.jpg`

**Narration:** Welcome. Today we learn machine learning the way it is actually done in research: on one real project, from raw data to a trained model. The project is about a wastewater treatment plant. Bacteria clean the water, and they need oxygen, which we supply by blowing air. More air means cleaner water but more energy. Learning to control that trade-off is what our models try to do.

### Slide 2 · What we will do in 90 minutes
- The plant and the problem; what a model is
- Collect and clean the data (notebook 1)
- Prepare it: transform, scale, windows, split (notebook 1)
- Choose a model; inside a network, layer by layer; counting parameters
- Train it: loss, gradient descent, the training loop, overfitting (notebook 2)
- Reinforcement learning in brief (notebook 3)
- Image: `stills/slide02.jpg`

**Narration:** Each topic follows the same rhythm. A few short slides, then you run the same step yourself in a notebook. Every step points to the file in the repository that does it for real, so after today you can read the project's code. You need a laptop with Python; no GPU is needed.

### Slide 3 · What the project is about
- The plant: BSM1, five tanks and a settler
- We measure oxygen, nitrate and what flows in; we control the air and a recycle pump
- Goal: oxygen at 2 mg/L and nitrate at 1 mg/L, without wasting energy
- Four ML tasks: forecasting (regression), early warning (classification), a digital twin (sequence model), control (reinforcement learning)
- Image: `stills/slide03.jpg`

**Narration:** The project has four machine-learning tasks, and each is a different kind of learning. Forecasting the inflow is regression. Warning before ammonium breaks the legal limit is classification. Predicting how the plant responds is a sequence model. Deciding how much air to blow is reinforcement learning. Today we follow the forecasting task from start to finish, then see how the others differ.

### Slide 4 · How the project was built: one script per step
- Plant and data → reproduce the three papers → build our controller and baselines → test → report
- Each box is one script in experiments/, with the command that runs it and the file it writes
- Image: `stills/slide04.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before the details, here is the whole project on one page. Each box is one script. We first rebuilt the official plant and data, then reproduced the three published controllers exactly as printed, then built our own controller and two strong baselines, tested everything, and wrote the report from the result files. Click any box to see the command and what it found.

### Slide 5 · What is a model?
- A model is a function with knobs: ŷ = w·x + b
- The knobs w and b are the parameters
- Loss: one number for how wrong the model is
- Training: turn the knobs a little in the direction that lowers the loss, and repeat
- A neural network is the same with thousands of knobs
- Image: `stills/slide05.jpg`

**Narration:** Everything today builds on this one picture. A model is a function with knobs. Here the function is a straight line with two knobs, its slope and its offset. The red lines show how wrong it is, and their average squared length is the loss. Training means turning the knobs a little in the direction that makes the loss smaller, again and again. A neural network is exactly this, only with thousands of knobs.

### Slide 6 · Where the data comes from
- Official IWA influent files: 14 days, every 15 minutes, 1 345 rows, 15 columns
- Three weathers: dry, rain and storm
- Week 1 is the same dry week in all three files: a trap for splitting
- Image: `stills/slide06.jpg`

**Narration:** Our data is what flows into the plant: the flow rate and thirteen concentrations, every fifteen minutes, for two weeks. There are three weather scenarios. Notice that the first week is identical in all three files. If we are careless when we split the data, we will test on what we trained on. Keep that in mind.

### Slide 7 · Load it once, record where it came from, cache it
- Download from a pinned version (a commit hash), so the bytes never change
- Keep only the rows that are data; spreadsheets carry titles and units
- Cache in a fast format
- Don't copy data you may not redistribute
- Code: `src/wwtp/official/plant.py`, line 30
- Image: `stills/slide07.jpg`

**Narration:** Four habits for collecting data. Pin the exact version you used, so anyone can rebuild your dataset. Keep only the real data rows. Save a fast copy so you don't parse the spreadsheet every time. And respect the licence: these files are downloaded at run time, not copied into the repository.

### Slide 8 · Cleaning: look before you learn
- Text rows: drop them, but write the units down
- Missing values: interpolate short gaps, drop long ones
- Duplicate timestamps; check the time step is constant
- Spikes: flag against a rolling median
- Physics check: inert COD should be diluted by rain; the project corrects it
- Write every correction down
- Image: `stills/slide08.jpg`

**Narration:** Before learning anything, look at the data. Real sensor data has missing values, duplicates, spikes and stuck sensors. Statistics finds most of these. But one error in this project was only found with physics: one concentration stayed constant while rain doubled the flow, which is impossible, because rain dilutes it. The project corrects it and documents the correction. Every correction you make should be written down.

### Slide 9 · Put everything on one time grid
- Sensors sample every 15 minutes; the controller acts every 45 seconds
- Between samples the value is held constant: 20 control steps per sample
- Image: `stills/slide09.jpg`

**Narration:** Different instruments work at different rates. The influent is sampled every fifteen minutes but the controller acts every forty-five seconds, so each sample is held for twenty control steps. Always put all your signals on one time grid before you build a dataset.

### Slide 10 · How the project loads and corrects the official data
- The real function, load_influent in src/wwtp/official/plant.py, block by block
- Held-out data, cached raw table, the column layout, the S_I fix, derived columns
- Image: `stills/slide10.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is the project's own code for what we just discussed. Step through it: the raw table comes from the cache, the columns are laid out the way the plant simulator expects, and one line applies the rain-dilution fix.

### Slide 11 · Collect and clean, cell by cell
- Real notebook cells with the outputs of a real run
- Pin the source, look before you parse, standard checks, repair a damaged copy
- Image: `stills/slide11.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Before you start, here is the notebook running. Each cell has its real output, and the panel explains the idea behind it.

### Slide 12 · Your turn: notebook 1, parts A and B (12 minutes)
- Download, open, name the columns, check the time step, plot the three weathers
- Repair a deliberately damaged copy; apply the dilution fix; resample
- Image: `stills/slide12.jpg`

**Narration:** Now open notebook one. You will load the raw file and see why reading it carelessly turns every number into text. Then you will repair a copy we damaged on purpose and measure how good your repair is, because here we know the truth.

### Slide 13 · Transform and scale the inputs
- Log for flows and loads, so storms don't dominate
- Standardise: z = (x − mean) / std
- Compute mean and std on the training data only, and reuse them everywhere
- Save the scaler with the model
- Image: `stills/slide13.jpg`

**Narration:** Networks learn best when every input sits around zero with a similar spread. Flow is around twenty thousand, ammonium around thirty; without scaling, flow would dominate. We take the log, then standardise. One rule matters most: compute the mean and standard deviation on the training data only. Otherwise information from the test set leaks into training.

### Slide 14 · Turn one long series into many examples
- Slide a window along the series: the input is the past 24 hours, the target is the next 2 hours
- The input ends exactly where the target begins: the model only sees the past
- Result: X with shape (N, 96, 3) and y with shape (N, 8)
- Image: `stills/slide14.jpg`

**Narration:** A model learns from examples, so we cut the long series into many of them. A window slides along the data. The blue part, the last twenty-four hours, is the input; the orange part, the next two hours, is the target. Stack them all and you get a three-dimensional array: number of examples, time steps, channels. Remember this shape: it decides the model.

### Slide 15 · Split by time, not at random
- Neighbouring windows share 95 % of their samples
- A random split puts near-copies of training examples in the test set
- Train on the past, validate later, test last
- The project: week 1 versus week 2, separate simulated plants, a test protocol written before the test was run
- Image: `stills/slide15.jpg`

**Narration:** With time series, never split at random. Neighbouring windows overlap almost completely, so a random split hides copies of the training data in the test set, and the score looks far better than reality. Split by time instead. The project went further: it wrote down its final test before running it, and reported the result whichever way it went.

### Slide 16 · Prepare, cell by cell
- The physics fix, scaling with training data only, windows and shapes, one example in physical units
- Image: `stills/slide16.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The preparation steps in code. Watch the shapes: ninety-six steps, three channels, eight targets.

### Slide 17 · Your turn: notebook 1, part C (10 minutes)
- Inputs: flow, ammonium, COD; target: the next 8 flow values
- Train = dry plus rain week 2; validation = storm 1; test = storm 2
- Scale with training statistics, cut windows, check shapes, save
- Image: `stills/slide17.jpg`

**Narration:** Prepare the dataset yourself. Notice that training contains no storm, but the test is a storm. Think about what that means for any model before you train one.

### Slide 18 · The input and output decide the model
- Table data: linear models, gradient-boosted trees, MLPs
- Sequences: convolutional networks, recurrent networks, transformers
- A decision with only a reward: reinforcement learning
- Output decides the last layer and the loss: number → MSE; class → cross-entropy; probability → binary cross-entropy; range → quantile loss; action → tanh
- Image: `stills/slide18.jpg`

**Narration:** How do you choose a model? Start from the shapes. If your input is a sequence, use a model that knows about time order. If your output is a number, use a linear last layer and a squared or absolute error. If it is a probability, use a sigmoid and cross-entropy. The six models of this project each follow these rules. The amount of data then limits how big the model can be.

### Slide 19 · Inside a neural network, layer by layer
- Each neuron: weighted sum of its inputs plus a bias, then an activation (ReLU: negatives become zero)
- A whole layer at once: h = ReLU(W·x + b), one matrix multiplication
- Training changes only the weights and biases
- Image: `stills/slide19.jpg`

**Narration:** Let's open the box. Three inputs come in: flow, ammonium and COD. Each neuron of the first layer multiplies every input by its own weight, adds them up, adds a bias, and then applies ReLU, which turns negative values into zero. The next layer does the same with these results, and the last neuron gives the forecast. That's all a network does. Training only changes the numbers on the connections.

### Slide 20 · Counting parameters
- A dense layer from n to m units has n·m weights plus m biases
- Our MLP, 288 → 64 → 64 → 8, has 23 176 parameters
- Compare with your number of examples: about 1 900 here
- Far more parameters than examples means the model can memorise
- Image: `stills/slide20.jpg`

**Narration:** Count your parameters and compare them with your data. Our small network has about twenty-three thousand parameters for nineteen hundred examples. The reinforcement-learning agent of one paper has sixty-seven thousand. When parameters far outnumber examples, the model can simply memorise the training set, and we will see that happen in the notebook.

### Slide 21 · A causal convolution
- The same small filter slides along time
- Causal: it only looks at the past (padding on the left)
- Dilation 1, 2, 4… doubles how far each layer looks back
- Image: `stills/slide21.jpg`

**Narration:** For sequences, a convolution slides one small filter along time and reuses the same weights everywhere, so a long input needs few parameters. Causal means it never looks at the future. With dilations, each layer looks twice as far back, so a few layers cover a whole day. This is the building block of the project's forecaster.

### Slide 22 · A recurrent network carries a memory
- The same cell at every step reads the new input and the previous memory
- The digital twin uses one: its memory stands for what no sensor measures, like the bacteria
- Image: `stills/slide22.jpg`

**Narration:** A recurrent network reads the sequence one step at a time and carries a memory forward. In the project's digital twin, that memory stands in for things no sensor measures, like how many bacteria are in the tanks.

### Slide 23 · Shapes, baselines and models, cell by cell
- Baselines, a linear model, an MLP with its parameter count
- One example followed layer by layer; a causal CNN and the proof that it never sees the future
- Image: `stills/slide23.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Here the concepts become code. The layer-by-layer cell prints the shape after every layer of the real network, and the causal network proves in two lines that it cannot see the future.

### Slide 24 · The loss says what "wrong" means
- MSE punishes big errors and gives the mean; MAE is robust and gives the median
- Pinball loss gives quantiles: a range instead of a single number
- Cross-entropy for classes and yes/no events
- Image: `stills/slide24.jpg`

**Narration:** The loss is your definition of wrong, and choosing it is a design decision. Squared error punishes big mistakes. Absolute error is robust to outliers. The pinball loss lets a model predict a range, which tells an operator how sure the forecast is.

### Slide 25 · Gradient descent
- w ← w − learning rate × gradient
- Too small: barely moves; too large: overshoots and diverges
- Backpropagation computes every gradient in one backward pass
- In practice: Adam or AdamW, a learning-rate schedule, gradient clipping
- Image: `stills/slide25.jpg`

**Narration:** Training walks downhill on the loss. The learning rate is the step size. Too small and nothing happens; too large and the model overshoots and blows up. Backpropagation computes the slope for every weight at once. In practice we use the Adam optimiser with a schedule and gradient clipping, as every training file in the repository does.

### Slide 26 · The training loop
- Batch → forward → loss → backward → update; repeat
- After each epoch, check the validation loss
- Keep the best epoch: when validation rises while training falls, the model is overfitting
- Image: `stills/slide26.jpg`

**Narration:** Every training script in the world is this loop. Take a batch, predict, compute the loss, compute the gradients, update the weights. After each pass through the data, check the validation loss and keep the best model. When training loss keeps falling but validation loss rises, the model has started to memorise.

### Slide 27 · Training, overfitting and the test, cell by cell
- The training loop; both loss curves; overfitting on purpose
- The test set used once; saving the model with its scaler
- Image: `stills/slide27.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The training loop in real code, then overfitting on purpose: four hundred thousand parameters on two hundred examples. Finally the test, used once, where the simplest model wins.

### Slide 28 · A model is only good compared with something simple
- Persistence, "same time yesterday", a linear model, gradient-boosted trees
- Report errors in physical units, on the test set, over several seeds
- In this project: a published learning controller did nothing beyond its PID, and a carefully tuned PID matched our own learned controller
- Image: `stills/slide28.jpg`

**Narration:** Always compare with something simple. If a free rule like "the next two hours equal now" is as good as your network, the network isn't earning its keep. This project learned it the hard way: a well-tuned classical controller matched the learned one. That is not a failure, it is a result, and it belongs in the report.

### Slide 29 · Your turn: notebook 2 (20 minutes)
- Baselines, a linear model, an MLP and a causal CNN
- Print every layer's output shape; write the training loop
- Overfit on purpose, then fix it
- Test once; save the model with its scaler; reload and predict
- Image: `stills/slide29.jpg`

**Narration:** Train the models yourself and fill in the table of parameters and test errors. Watch which model wins. On this small dataset the answer may surprise you.

### Slide 30 · Reinforcement learning: no correct answer to copy
- Supervised: a fixed dataset with the right answer
- Reinforcement: data collected by acting, and a reward that often comes late
- The agent observes the state, acts, the plant responds, a reward comes back
- The reward is the specification: the agent does what you wrote
- Image: `stills/slide30.jpg`

**Narration:** For control there is no right answer to copy. The agent acts, the plant responds, and a reward says how good that was. The reward is where you write down what you want, and the agent follows it literally. One paper's agent saved energy but broke the ammonium limit far more often, because that is what its reward allowed.

### Slide 31 · The parts of an actor-critic agent
- Environment: reset and step
- Actor: state → action; critic: (state, action) → expected future reward
- Replay buffer, target networks, exploration noise
- The critic is trained like a regression against a target it makes for itself
- Image: `stills/slide31.jpg`

**Narration:** Modern agents such as DDPG and SAC have two networks. The actor decides the action; the critic estimates how much reward will follow. The critic is trained exactly like the regression you just did, only its target comes from its own estimates. Same loop, same optimiser.

### Slide 32 · Reward, DDPG update and SAC-IL update, in the project's code
- The reward of paper 2 and what it leaves out
- DDPG: replay, Bellman target, critic regression, actor, soft targets
- SAC-IL: mixed batches, entropy, an imitation term that never fades out
- Image: `stills/slide32.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Now the reinforcement-learning code of the project. The reward is two lines, and what it leaves out explains what the agent did. The critic update is just a regression. Our SAC-IL keeps imitating the expert while it learns.

### Slide 33 · Reinforcement learning, cell by cell
- Environment, expert, imitation, Q-learning, comparison, reward change
- Image: `stills/slide33.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** The RL notebook running, cell by cell, with its real results.

### Slide 34 · Start from the expert, then improve, behind a safety gate
- 1. Run the expert PID and record what it does
- 2. Imitation: copy it with a network (supervised learning)
- 3. Improve with reinforcement learning
- 4. Safety gate: outside the training range, fall back to the expert
- Image: `stills/slide34.jpg`

**Narration:** Learning from scratch on a real plant means weeks of bad decisions. So the project's controller first copies the existing expert controller, which is ordinary supervised learning, and only then improves with reward. A safety gate hands control back to the expert whenever conditions look unfamiliar.

### Slide 35 · Papers 1 and 3 exactly as printed, and why they don't learn
- MAACC: incremental action network, critic, the printed critic update, a learning signal of about 10⁻⁴
- IOQL: 50 critic and 60 actor steps per period on one sample
- Image: `stills/slide35.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** This is reproduction in practice: the published equations, line by line, in code. Paper 1's learning signal is so small that the controller never leaves its PID. Paper 3 repeats so many updates on one sample that it becomes unstable.

### Slide 36 · ILC, the control step, the safety gate and the training script
- Daily ILC: learn each night what the PID had to do, and do it earlier
- One control period: PID plus learned correction, bumpless at midnight
- The safety gate; the training script from expert to evaluation
- Image: `stills/slide36.jpg`
- Interactive in the HTML deck: use the live deck for this slide.

**Narration:** Our controller, in four files. The daily learning control, the control step that combines PID, feedforward and the learned correction, the safety gate that hands control back to the expert, and the training script from start to end.

### Slide 37 · Your turn: notebook 3 (12 minutes)
- A one-tank oxygen environment driven by the real daily load
- The PID, imitation, and Q-learning from reward alone
- Test on days never used for training; change the reward and see what happens
- Image: `stills/slide37.jpg`

**Narration:** Train an agent in seconds. Compare it with the expert and with the expert plus one line of physics. Then make energy expensive in the reward and watch what the agent does to the oxygen.

### Slide 38 · What this project teaches about doing ML research
- Look at the data first
- The input and output shapes choose the model; the data limits its size
- Split by time, scale on training data only, never tune on the test
- Baselines first; reproduce before you improve
- Several seeds; write the test down before you run it
- Save the model with everything it needs
- Image: `stills/slide38.jpg`

**Narration:** To finish, eight habits from this project. Look at your data. Let the shapes choose the model. Split honestly. Beat simple baselines. Reproduce published work before claiming to improve it. Use several seeds. Write your test down before you run it. And save your model with everything it needs to be used again. Everything we showed is in the repository. Questions?


# EM Synthesis Project

Our goal is to synthesize a programmatic state machine policy from time-series data while simultaneously inferring a set of high-level action labels.

Our system is a *discrete-time Markov process* defined by:
   - a **high-level action space** $H$ = a set of discrete high-level actions $h \in H$
     - Ex: $h \in$ {ACC, DEC, CON}
   - a **low-level action space** $L$ = a continuous domain of low-level actions $l \in L$: controlled joystick directives, motor inputs, etc.
     - Ex: $l = accel \in \mathbb{R}$, where $a$ is the acceleration
   - a **observed state space** $O$ = a continuous domain of observed variables $o \in O$.
     - Ex: $o = (pos, vel, accel) \in \mathbb{R}^3$
   - an **action-selection policy (ASP)** $\pi: H \times O \rightarrow H$ that maps the current high-level action and the current observed variables to the next high-level action
   - a **motor model** $\phi: H \rightarrow L$ that maps the current high-level action to the current low-level action

---
## Overall problem formulation:
### Inputs
We define a trajectory $\tau$ over a time period $T$ as a function $\tau : T \rightarrow H \times O$, which can be thought of as giving the value of the system's state $(h_t, o_t)$ at each time step $t=0...T$.

We know the problem domain $H, L, O$, as well as the motor model $\phi$. We are given a set of **demonstrations**, which are defined simply as trajectories with the high-level labels missing, i.e. $o_t$ for $t = 0...T$.

### Outputs
We would like to:
1. Infer the values of the high-level actions in the demonstrations ($h_t$)
2. Synthesize an ASP that is maximally consistent with the demonstrations ($\pi^*$)

---
## Dependencies & Setup
See **pips/**. No further dependencies are required. (scipy?)

---
# How to run
To get the project running, you will need to make the following changes:

- in **domain.h**, define your high-level action space, low-level action space, and state space.
- in **robot.h**, define your motor model.
- in **pips/**, follow the instructions to define the desired operations and the dimensions of each observed variable.
- optionally, tune the desired hyperparameters in **settings.h**.

If you need to simulate your own demonstrations, you can also use our interface to:
- Define the ground-truth ASP and the physics model in **robot.h**.
- Set the desired demonstration robot(s) in **simulation/robotSets.h**.

Then, you can use *make* commands to run the project:
- **make** to build the project
- **make em** to run the full EM Synthesis algorithm, including simulating demonstrations
- **make emng** to run the EM Synthesis algorithm, without simulating demonstrations
- **make plt** to plot the algorithm outputs and store them into *synthesis/plots/*
- **make clean, make clear_data, make purge** to delete all build files, to clear all data/plots/trajectories, or both

Other *make* commands which are not required:
- **make gen** to run only the simulation
- **make pf** to run only the particle filter (E-step)
- **make settings** to compile settings
- **make snapshot** to archive current settings and output files to a given folder

## Project Organization
This project is roughly split into the following components:

- **simulation/** - simulated demonstrations for a 1D vehicle simulation
- **highway-env/** - gathering demonstrations for a lane-merging, 2D highway environment
- **particleFilter/** (expectation step) - runs a particle filter to get a set of most likely high-level actions
- **pips/** (maximization step) - runs a program synthesizer to generate the program that is maximally consistent with the given high-level actions
- **synthesis/** - runs the EM-loop, alternating between expectation and maximization steps
- **domain.h** - defines the problem domain, including the spaces $H$, $L$, and $O$
- **robot.h** - defines world models, including the simulation ASP, motor model, and physics model
- **settings.h** - consolidated list of hyperparameters and settings
- **utils.h** - useful functions for general use
- **translateSettings.cpp** - converts settings.h into a text file (settings.txt) for easy Python interpretation
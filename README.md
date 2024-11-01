**This is the forked repo I worked with during my time as Research Assitant Intern at Fudan University, it includes comments I wrote to further understand and document the project.**
**The following is the original README:**

# PLUNDER: Probabilistic Program Synthesis for Learning from Unlabeled and Noisy Demonstrations

![](2D-highway-env/snapshots/example_snapshot/asp_8.gif)

Our goal is to synthesize a programmatic state machine policy from time-series data while simultaneously inferring a set of high-level action labels. 

---
## Related Resources
PLUNDER (codebase): [https://github.com/ut-amrl/plunder](https://github.com/ut-amrl/plunder)

AMRL Google Drive (presentations and videos): [https://drive.google.com/drive/folders/1QaKtIvmKhZjxIwY9ANSPpjYl0teoNW5S?usp=share_link](https://drive.google.com/drive/folders/1QaKtIvmKhZjxIwY9ANSPpjYl0teoNW5S?usp=share_link)

Publication: [https://arxiv.org/abs/2303.01440](https://arxiv.org/abs/2303.01440)

Project Website: [https://amrl.cs.utexas.edu/plunder](https://amrl.cs.utexas.edu/plunder)

---
## Description

Our system is a *discrete-time Markov process* defined by:
   - an **action space** $A$ = a set of discrete action labels $a \in A$
     - Ex: $a \in$ {ACC, DEC, CON}
   - a **low-level observation space** $Z$ = a continuous domain of low-level observations $z \in Z$: controlled joystick directives, motor inputs, etc.
     - Ex: $z = acc \in \mathbb{R}$, where $acc$ is the acceleration
   - a **state space** $S$ = a continuous domain of constants or variables $c, y \in S$.
     - Ex: $c = accMax \in \mathbb{R}, y = pos \in \mathbb{R}$
   - an **action-selection policy (ASP)** $\pi: A \times S \rightarrow A$ that maps the current action label and the current state to the next action label
   - an **observation model** $O: A \rightarrow distr(Z)$ that maps discrete action labels to a distribution over low-level observations via discrete motor controllers

---
## Overall problem formulation:
### Inputs
We know the problem domain $A, Z, S$, as well as the observation model $O$. We are given a set of **demonstrations**, which are defined simply as trajectories with the action labels missing, i.e. $s_{1:t}$ and $z_{1:t}$.

### Outputs
We would like to:
1. Infer the values of the action labels in the demonstrations ($a_{1:t}$)
2. Synthesize an ASP that is maximally consistent with the demonstrations ($\pi^*$)

---
## Dependencies & Setup
See **pips/**. 
In addition, this project requires Scipy: https://scipy.org/install/.
If you wish to run the highway environment yourself, you'll need highway-env and its dependencies: https://highway-env.readthedocs.io/en/latest/installation.html
If you wish to run the robotic arm environment yourself, you'll also need panda-gym and its dependencies: https://panda-gym.readthedocs.io/en/latest/index.html

---
# How to run examples
We have provided five example tasks: 1D-target, 2D-highway-env, 2D-merge, panda-pick-place, and panda-stack.

To run these tasks:
1. Go into the Makefile and set the variable *target_dir* to the desired folder (default set to 1D-target).
2. Run **make** to build the project.
3. Run **make em** (for 1D-target) or **make emng** (for the other tasks) to run PLUNDER.

Please see each of these folders for an extended usage guide.

---
# Further configuration
To setup a custom environment, you will need to do the following:
- Create a new folder to house your problem domain. 
- In that directory, create the files **domain.h, robot.h, settings.h,** and **emdips_operations.json**. 
- In **domain.h**, define your action space, observation space, and state space.
- In **robot.h**, define your observation model.
- In **settings.h**, tune the desired parameters and I/O paths.
- In **emdips_operations**, define your desired operations (plus, minus, times, etc). See *pips/* for general tips and guidelines for defining operations and a list of existing operations.

If you need to simulate your own demonstrations, you can also use our interface to:
- Define the ground-truth ASP and the physical simulation model in **robot.h**.
- Set the desired demonstration initial states in another file **robotSets.h**.

An example setup is defined in *1D-target*; it may be easier to copy paste that folder and work from there.

Then, you can use *make* commands to run the project:
- **make** to build the project. (Go into the Makefile and set the variable *target_dir* to the desired folder, then run *make*.)
- **make em** to run the full EM Synthesis algorithm, including simulating demonstrations
- **make emng** to run the EM Synthesis algorithm, without simulating demonstrations
- **make plt** to plot the algorithm outputs and store them in png format
- **make clean, make clear_data, make purge** to delete all build files, to clear all data/plots/trajectories, or both
- **make snapshot** to archive current settings and output files to a given folder

Other *make* commands which are not commonly used alone:
- **make gen** to run only the simulation
- **make pf** to run only the particle filter (E-step)
- **make settings** to compile settings

---
## Project Organization
This project is roughly split into the following components:

- **simulation/** - for simulating demonstrations given a ground-truth ASP
- **particleFilter/** (expectation step) - runs a particle filter to get a set of most likely action labels
- **pips/** (maximization step) - runs a program synthesizer to generate the program that is maximally consistent with the given action labels
- **synthesis/** - runs the EM-loop, alternating between expectation and maximization steps
- **system.h** - fully defines the discrete-time Markov process given *domain.h* and *robot.h*
- **utils.h** - useful functions for general use
- **includes.h** - all include statements for tidiness
- **translateSettings.cpp** - converts settings.h into a text file (settings.txt) for easy Python interpretation

#pragma once

#include "includes.h"

using namespace std;

namespace SETTINGS {

    // I/O parameters
    const string SIM_DATA = "sim/data";                         // Generated data from simulation, contains HA, LA, and observed state sequences
    const string GEN_ASP = "out/asp_iter";                      // ASPs generated by EMDIPS
    const string TRAINING_TRAJ = "out/training_traj/iter";                  // Trajectories (high-level action sequences) generated by particle filter
    const string TESTING_TRAJ = "out/testing_traj/iter";              // Trajectories generated from ASPs
    const string VALIDATION_TRAJ = "out/validation_traj/iter";  // Trajectories generated from ASPs, 
    const string PLOT_PATH = "plots/";                          // Plots
    const string GT_ASP_PATH = "gt_asp/";                       // Ground truth ASP
    const string LOG_OBS_PATH = "out/log_obs";                  // Cumulative log observation likelihood across all runs
    const string PCT_ACCURACY = "out/pct_acc";                  // Percent accuracy compared to the given trajectory
    
    // General Configuration
    const bool DEBUG = true;
    const double EPSILON = 10E-10;
    const int PRECISION = 10;
    const int TRAINING_SET = 10;                 // number of robots to train on
    const int VALIDATION_SET = 30;              // number of total robots in the validation set

    // Simulation parameters
    const double T_STEP = .1;               // time step (s)
    const double T_TOT = 12.5;                // total time (s) per simulated scenario
    const double GEN_ACCURACY = 1.0;        // probability of a correct high-level transition

    // EM Loop parameters
    const int NUM_ITER = 20;                    // number of iterations in the expectation-maximization loop
    const int SAMPLE_SIZE = 100;                 // number of trajectories to process then pass into EMDIPS, per robot
    const double POINT_ACCURACY = 0.9;          // probability of a correct (ASP-consistent) high-level transition
    const int STRUCT_CHANGE_FREQ = 1;           // only enumerate over new program structures every n iterations, else tune parameters for previous best structure

    // Plot parameters
    const bool GT_PRESENT = true;
    const int PLOT_TIME = 125;                                          // Maximum time step plotted

    // Optimization parameters
    const int OPT_METHOD = 1;               // Optimization method:
                                            // 0: local (BFGS)
                                            // 1: local (L-BFGS-B)
                                            // 2: basin hopping
                                            // 3: dual annealing
                                            // 4: DIRECT
    const bool ENUMERATE_SIGNS = false;      // Equivalent to enumerating over > and <
    const bool PRINT_DEBUG = false;         // Extra debugging info
    const int INITIAL_VALUES = 4;           // Initial values for x_0: 0 = all zeros, 1 = average, >1 = do all of the above, then enumerate over random initial guesses (use this to specify how many)
    const int BATCH_SIZE = 16;               // Number of programs to optimize in parallel
    const int NUM_CORES = 4;                // Number of cores to use per program: NUM_CORES * BATCH_SIZE = total number of cores used at once
    const double INIT_ALPHA = 0.0;          // starting slope
    const double OUTLIER_MAX = 20;          // Max negative log likelihood that an example can contribute to the total log likelihood
    const int MAX_ITER = 200;               // Max number of iterations of a single optimization run

    const double PROG_COMPLEXITY_LOSS = 0.005;      // adds L1 loss ( AST.size * PROG_COMPLEXITY_LOSS )
    const double ALPHA_LOSS_UPPER = 0.01;         // adds L2 loss ( alpha^2 * ALPHA_LOSS_UPPER )

    const int EX_SAMPLED = 1000;                // Number of examples to be optimized over

    // EMDIPS parameters
    const int PROG_ENUM = 128;                          // Number of programs to enumerate and optimize per iteration
    const bool USE_SAFE_TRANSITIONS = false;          // "safe" transitions (only allow user-specified transitions)
    const int BASE_FEAT_DEPTH = 3;

    // Particle filter parameters
    const int NUM_PARTICLES = 2000;                                // number of particle trajectories created to represent the distribution
    const double RESAMPLE_THRESHOLD = 10.0;                           // higher = more resampling
    double TEMPERATURE = 1;                                         // Initial observation likelihood strength
    const double TEMP_CHANGE = 0;                                // TEMPERATURE decreases linearly by this much each iteration
}
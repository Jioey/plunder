#pragma once

#include <string>

using namespace std;

// Robot parameters
const int robotTestSet = 0;         // which robots to use (0-2)
const int numRobots = 10;           // number of robots (depends on robot test set)
const int model = 3;                // which ASP to use
const double meanError = 0.0;       // low-level action error
const double stddevError = 0.1;     // low-level action error standard deviation
const double laChangeSpeed = 0.5;
const double switchingError = 0.0;  // additional low-level action error standard deviation while transitioning

// I/O parameters
const string stateGenPath = "accSim/out/data";                  // Generated data from simulation, contains HA, LA, and observed state sequences
const string aspPathBase = "synthesis/out/asp";                 // ASPs generated by EMDIPS
const string trajGenPath = "synthesis/out/examples/pf";         // Trajectories (high-level action sequences) generated by particle filter
const string altPath = "synthesis/out/states/iter";             // Trajectories generated from ASPs
const string operationLibPath = "pips/ops/emdips_test.json";    // Operation library for EMDIPS
const string plotGenPath = "synthesis/plots/";                  // Plots
const string gt_asp = "synthesis/gt_asp/";                      // Ground truth ASP

// Simulation parameters
const double T_STEP = .1;               // time step (s)
const double T_TOT = 15;                // total time (s) per simulated scenario
const double genAccuracy = 1.0;         // probability of a correct high-level transition
const double activationMinAcc = 0.0;    // Minimum acceleration (acceleration will not go below this value, excluding 0)
const double distErrorMean = 0.0;       // Perception error for distance
const double distErrorDev = 0.00;
const double velErrorMean = 0.0;        // Perception error for velocity
const double velErrorDev = 0.00;

// EM Loop parameters
const int numIterations = 10;             // number of iterations in the expectation-maximization loop
const int sampleSize = 10;                // number of trajectories to process then pass into EMDIPS, per robot
const bool usePointError = true;          // point error: random transitions to a new high-level action
const double pointAccuracy = 0.95;        // probability of a correct (ASP-consistent) high-level transition
const int structuralChangeFrequency = 1;  // only enumerate over new program structures every n iterations, else tune parameters for previous best structure
const bool hardcode_program = false;      // if true then only consider single hardcoded program structure

// Optimization parameters
// TODO: move to here; find a way to efficiently retrieve them from here to the Python script

// EMDIPS parameters
const int window_size = 11;
const int sampling_method = 1;                    // 1 for default window sampling, or 2 for custom
const int feature_depth = 3;
const int sketch_depth = 2;
const float max_error = 0.03;                     // Target log likelihood threshold to stop enumeration early
const int batch_size = 8;
const int max_examples = 40;
const int programs_enumerated = 7;
const bool useSafePointError = true;              // "safe" transitions (only allow user-specified transitions)

// Plot parameters
const int particlesPlotted = 100;
const int timeStepsPlot = 150;

// Particle filter parameters
const int numParticles = 20000;                                  // number of particle trajectories created to represent the distribution
const int numTrajectories = max(sampleSize, particlesPlotted);  // number of particle trajectories sampled to be fed into the maximization step
const float resampleThreshold = 1.0;                            // higher = more resampling
const double pf_stddevError = 0.1;
const float obsLikelihoodStrength = 1.0;                        // lower = stricter observation likelihood
const int end_pf_err = 0;                                       // ignores last n timesteps because they didn't have a chance to get resampled
const bool useSimplifiedMotorModel = false;                      // Use simulation motor model or a simplified version

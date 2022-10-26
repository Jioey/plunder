#pragma once

#include <string>

using namespace std;

// Robot parameters
const int robotTestSet = 0;         // which robots to use (0-2)
const int numRobots = 15;           // number of robots (depends on robot test set)
const int model = 0;                // which ASP to use
const double meanError = 0.0;       // low-level action error
const double stddevError = 0.5;     // low-level action error standard deviation

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
const double T_TOT = 30;                // total time (s) per simulated scenario
const double genAccuracy = 1.0;         // probability of a correct high-level transition
const double activationMinAcc = 0.2;    // Minimum acceleration (acceleration will not go below this value, excluding 0)
const double distErrorMean = 0.0;       // Perception error for distance
const double distErrorDev = 0.00;
const double velErrorMean = 0.0;        // Perception error for velocity
const double velErrorDev = 0.00;

// EM Loop parameters
const int numIterations = 10;           // number of iterations in the expectation-maximization loop
const int sampleSize = 10;              // number of trajectories to pass into EMDIPS
const bool usePointError = true;        // point error: random transitions to a new high-level action
const double pointAccuracy = 0.95;       // probability of a correct (ASP-consistent) high-level transition
const int structuralChangeFrequency = 3;
const bool hardcode_program = true;

// EMDIPS parameters
const int window_size = 5;
const int feature_depth = 3;
const int sketch_depth = 2;
const float max_error = 0.3;               // Target threshold
const int batch_size = 8;
const int max_examples = 500;
const int programs_enumerated = 1;

// Plot parameters
const int particlesPlotted = 50;
const int timeStepsPlot = 500;

// Particle filter parameters
const int numParticles = 20000;                                  // number of particle trajectories created to represent the distribution
const int numTrajectories = max(sampleSize, particlesPlotted);  // number of particle trajectories sampled to be fed into the maximization step
const float resampleThreshold = 0.2;                            // higher = more resampling
const double pf_stddevError = 0.5;
const float obsLikelihoodStrength = 0.8;

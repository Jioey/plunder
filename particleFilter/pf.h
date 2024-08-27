#pragma once

#include "robot.h"

using namespace std;
using namespace SETTINGS;

static uint resampCount = 0; // Debug

typedef HA init();
typedef double obs_likelihood(State, Obs, double);

// ----- Helper Functions ---------------------------------------------

// Metric to calculate "effective" particles
double effectiveParticles(vector<double>& weights){
    double sum = 0;
    for(double x: weights){
        sum += x * x;
    }
    return 1 / sum;
}

// Resample particles given their current weights. 
// This method performs the resampling systematically to reduce variance.
vector<HA> systematicResample(vector<HA>& ha, vector<double>& weights, vector<int>& ancestor){
    resampCount++;
    int n = weights.size();

    if (labeler == GREEDY_HEURISTIC) {
        // Just take the particle with the highest weight
        HA bestHA = ha[0];
        double bestWeight = weights[0];
        for(int i = 0; i < n; i++) {
            if(weights[i] > bestWeight) {
                bestWeight = weights[i];
                bestHA = ha[i];
            }
        }

        vector<HA> haResampled(n, bestHA); // 找不到定义。。。但我猜是生成了n个bestHA组成的列表
        return haResampled;
    }

    vector<double> cumulativeWeights;
    vector<HA> haResampled;
    double runningSum = 0;
    for(double w: weights){
        runningSum += w;
        cumulativeWeights.push_back(runningSum);
    }

    double interval = 1.0 / (double) n;
    double pos = ((double) rand()) / RAND_MAX * interval; // Initial offset
    for(int i = 0, j = 0; i < n; i++){

        while(cumulativeWeights[j] < pos){
            j++;
        }
        haResampled.push_back(ha[j]);
        ancestor[i] = j;
        pos += interval;
    }
    return haResampled;
}

// ----- Particle Filter ---------------------------------------------

class ParticleFilter {
public:

    Trajectory state_traj;    // Observed state sequence (is the i-th trajectory of state_traj in em.cpp's expectation() function)
    vector<vector<HA>> particles;   // Gives the high-level trajectories of each particle
    vector<vector<int>> ancestors;  // Stores ancestors during resampling (unused, intended for non-greedy re-weighting method)

    asp* asp_pf;
    init* init_pf;
    obs_likelihood* obs_likelihood_pf;

    ParticleFilter(Trajectory& _state_traj, asp* _asp, init* _init, obs_likelihood* _obs_likelihood) : state_traj(_state_traj), asp_pf(_asp), init_pf(_init), obs_likelihood_pf(_obs_likelihood) {}

    // Run particle filter on num_particles particles
    // 利用当前的ASP预测下一个time step的action label，同时进行2000次
    // 根据一个RESAMPLE_THRESHOLD还会根据weights更改particles里的label
    // weights是根据一个liklihood function计算的
    double forwardFilter(int num_particles){

        // Initialization
        int N = num_particles; // defined as 2000 in settings.txt
        int T = state_traj.size(); // = number of lines in the data#.csv files

        vector<double> log_weights(N);
        vector<double> weights(N);
        double log_obs = 0.0;

        // initialize particles and ancestors
        for(int t = 0; t < T; t++){
            particles.push_back(vector<HA>(N)); // particles is a matrix where the rows represent time steps in a demonstration, and each row has N particles
            ancestors.push_back(vector<int>(N, 0));
        }
        
        // Sample from initial distribution
        for(int i = 0; i < N; i++){
            particles[0][i] = init_pf(); // initializes first row of particles as default HA (也就是0)
            ancestors[0][i] = i; // initializes first row of ancestors as its index
            log_weights[i] = -log(N); // initializes all log_weights as -ln(N)
            weights[i] = exp(log_weights[i]); // initializes all weights as e^(-ln(N))
        }

        // for every time step in demonstration
        for(int t = 0; t < T; t++){
            // Reweight particles
            for(int i = 0; i < N; i++){ // for each particle
                HA x_i = particles[t][i]; // set x_i as current particle
                Obs obs = state_traj.get(t).obs; // observation variables of the t-th time
                for(string each: LA_vars) { // for each Lower Action variables
                    obs.put(each, (t == 0) ? 0 : state_traj.get(t-1).get(each)); // update to its value one time step before, if it's not zero
                }

                // sets the particle's weight as the "log probability of observing given LA with a hypothesized high-level action"
                // the likelihood function is defined in system.h
                double log_LA_ti = obs_likelihood_pf(State (x_i, obs), state_traj.get(t).obs, TEMPERATURE);
                log_weights[i] += log_LA_ti;
            }

            // Normalize weights
            double log_z_t = logsumexp(log_weights); // exponentiates log_weights, sum them, then return the log of its sum
            double sum = 0.0;
            for(int i = 0; i < N; i++){
                log_weights[i] -= log_z_t;
                weights[i] = exp(log_weights[i]);
                sum += weights[i];
            }
            assert(abs(sum - 1.0) < EPSILON*N);

            // Update log observation likelihood
            log_obs += log_z_t;

            // Optionally resample when number of effective particles is low
            if(effectiveParticles(weights) < N * RESAMPLE_THRESHOLD){
                particles[t] = systematicResample(particles[t], weights, ancestors[t]); // updates particles[t] based on weights

                // Reset weights
                for(int i = 0; i < N; i++){
                    log_weights[i] = -log(N);
                    weights[i] = exp(log_weights[i]);
                }
            } else {
                // If not resammpled, ancestor for each particle is itself
                for(int i = 0; i < N; i++){
                    ancestors[t][i] = i;
                }
            }

            // Forward-propagate particles using provided action-selection policy
            if(t < T-1){
                for(int i = 0; i < N; i++){
                    // for each particle in the next time step, 
                    // set it as the asp prediction given: current predicted action & next time step's observations
                    particles[t+1][i] = asp_pf(State ( particles[t][i], state_traj.get(t+1).obs ));
                }
            } else { 
                // resample at last step to eliminate deviating particles
                particles[t] = systematicResample(particles[t], weights, ancestors[t]);
            }
        }
        return log_obs;
    }

    /**
     * @brief Retrieve high-level action sequences after running particle filter. Selects @param num_trajectories number of particles for each time step from the demonstraions and stores them in @param trajectories 
     * 
     * @param trajectories out-param: list of processed trajectories, which this function will append to
     * @param num_trajectories SAMPLE_SIZE, default set to 100
     * @return vector<vector<HA>> trajectories (the out-param itself)
     */
    vector<vector<HA>> retrieveTrajectories(vector<vector<HA>>& trajectories, uint num_trajectories){        
        if(particles.size() == 0){
            cout << "Run the particle filter first!" << endl;
            return vector<vector<HA>>();
        }
        assert(particles.size() == ancestors.size());

        int T = particles.size(); int N = particles[0].size();
        num_trajectories = min(num_trajectories, (uint) particles[0].size());

        // 当num_trajectories是2000的话，trajectories也就有2000行
        while(trajectories.size() < num_trajectories){
            trajectories.push_back(vector<HA>(T));
        }

        vector<uint> activeParticles;
        for(uint i = 0; activeParticles.size() < num_trajectories; i += (N / num_trajectories)){
            activeParticles.push_back(i);
        }

        // set i-th trajectory at t time as the t particle's active particle
        for(int t = T - 1; t >= 0; t--){
            for(uint i = 0; i < num_trajectories; i++){
                trajectories[i][t] = particles[t][activeParticles[i]]; //
            }
            if(t != 0){
                for(uint i = 0; i < num_trajectories; i++){
                    activeParticles[i] = ancestors[t][activeParticles[i]];
                }
            }
        }

        if(labeler == PERFECT) {
            for(uint i = 0; i < num_trajectories; i++){
                for(int t = 1; t < T; t++){ 
                    trajectories[i][t] = state_traj.get(t).ha;
                }
            }
        }

        return trajectories;
    }
};
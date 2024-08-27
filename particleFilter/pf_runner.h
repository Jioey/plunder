#pragma once

#include "pf.h"
#include "robotSets.h"
#include "settings.h"

using namespace std;
using namespace SETTINGS;

// ----- Markov System Parameters ---------------------------------------------

// Initial distribution
HA sampleInitialHA(){
    // Use default label
    return 0;
}

// ----- I/O ---------------------------------------------

// Read low-level action sequence and observed state sequence from file
void readData(string file, Trajectory& traj){
    ifstream infile;
    infile.open(file);
    string res;

    // header line
    getline(infile, res);
    istringstream iss_header (res);
    string comma;

    map<string, int> inds; // indicies (?)
    string cur_var;
    int count = 0;
    while(getline(iss_header, cur_var, ',')) {
        trim(cur_var);
        if(cur_var.substr(0, 3) == "LA.") // removes LA prefix if exists
            cur_var = cur_var.substr(3);

        inds[cur_var] = count; // --> inds = {(variable_name, id),...}
        count++;
    }


    // data lines
    while(getline(infile, res)){
        istringstream iss (res);

        // reads line into vals as doubles
        vector<double> vals;
        for(int i = 0; i < count; i++){
            double d; 
            iss >> d >> comma;
            vals.push_back(d);
        }

        // state creation
        State state;
        for(Var each: Obs_vars) {
            state.put(each.name_, vals[inds[each.name_]]);
        }

        if(inds.count("HA") > 0) { // if HA exists, set it in state
            state.set_ha(vals[inds["HA"]]);
        }

        traj.append(state);
    }
}

// Write high-level action sequences (trajectories) to file
void writeData(string file, vector<vector<HA>>& trajectories){
    ofstream outFile;
    outFile.open(file);

    for(vector<HA>& traj : trajectories){
        for(uint i = 0; i < traj.size(); i++){
            outFile << traj[i];
            if(i != traj.size() - 1){
                outFile << ",";
            }
        }
        outFile << endl;
    }

    outFile.close();
}

// ----- Particle Filter ---------------------------------------------
/** 
 * @brief Full trajectory generation with particle filter
 * 
 * @param trajectories out-param; The list storing post-filter trajectories (a list of action labels) 
 * @param N Number of particles to run the filter with
 * @param M Number of trajectories
 * @param traj The trajectory to input to the particle filter
 * @param asp Current synthesized ASP policy
 * @return double 
 */
double runFilter(vector<vector<HA>>& trajectories, int N, int M, Trajectory& traj, asp* asp){

    // Initialization
    srand(0);
    ParticleFilter pf (traj, asp, &sampleInitialHA, &obs_likelihood_given_model);
    resampCount = 0;

    cout << "Printing pre-filter trajectory 0..." << endl;
    for (int j = 0; j < traj.size(); j++)
    {
        cout << traj.get(j).to_string() <<  ", ";
    }
    cout << endl;    
    
    // Run particle filter
    double obs_likelihood = pf.forwardFilter(N);

    pf.retrieveTrajectories(trajectories, M);

    cout << "Printing post-filter trajectory 0..." << endl;
    for (int j = 0; j < trajectories[0].size(); j++)
    {
        cout << trajectories[0][j] << ", ";
    }
    cout << endl;
    

    if(DEBUG)
        cout << "resample count: " << resampCount << endl;

    return obs_likelihood;
}

/**
 * @brief Read input, run filter, write output
 * 
 * @param trajectories out param; A list of action labels based on the trajectories after the filter; initially is empty
 * @param N Number of particles to run the filter with
 * @param M Sample size (defined in settings.txt)
 * @param inputFile Path and name of the demonstration file, only read if @param traj is empty (应该是redundant的)
 * @param outputFile Path and name of the training trajectory output files 
 * @param traj Trajectory to process 
 * @param asp Current synthesized ASP policy
 * @return double 
 */
double filterFromFile(vector<vector<HA>>& trajectories, int N, int M, string inputFile, string outputFile, Trajectory traj, asp* asp){
    // Read input if traj is empty
    if(traj.size() == 0){
        readData(inputFile, traj);
    }

    // traj是打好标签的，trajectories是空的
    double obs_likelihood = runFilter(trajectories, N, M, traj, asp);

    // Write results (filter与sample完的action labels)
    writeData(outputFile, trajectories); 

    return obs_likelihood;
}
#include "particleFilter/pf_runner.h"
#include "simulation/generate.h"

using namespace std;
using namespace AST;
using namespace SETTINGS;
using Eigen::Vector2f;
using json = nlohmann::json;

vector<pair<string,string>> transitions;
vector<FunctionEntry> library;
vector<ast_ptr> roots;
PyObject* pFunc;

vector<float> loss;
vector<ast_ptr> solution_preds;
vector<ast_ptr> gt_truth;

namespace std {
    ostream& operator<<(ostream& os, const AST::ast_ptr& ast);
}

/**
 * @brief Convert transition to EMDIPS-compatible Example
 * 
 * @param ha Action label
 * @param state State's observations
 * @return Example 
 */
Example dataToExample(HA ha, Obs state){
    Example ex;

    // for each of the observation variables
    for(Var each: Obs_vars) {
        // if variable is a root 
        if(each.root_) {
            ex.symbol_table_[each.name_] = SymEntry((float) state.get(each.name_)); // add variable to example's symbol table with its value and type NUM (see definition in pips/src/ast.cpp)
        }
    }

    // adds action label name to the Example object as start_, with type STATE
    // (print(HA) is defined in utils.h, returns the string name of the action label)
    ex.start_ = SymEntry(print(ha)); 

    return ex; // returns the Example object
}

bool use_error = true;
// Runs EMDIPS-generated ASP
HA emdipsASP(State state){
    HA prev_ha = state.ha;
    Example obsObject = dataToExample(state.ha, state.obs); // converts input to Example data type

    // for the number of transitions (defined initially)
    for(uint i = 0; i < transitions.size(); i++){
        // when transitions[i] is relavent (i.e. prev_ha is the first of a transition and the transition is to another state)
        if(print(prev_ha) == transitions[i].first && transitions[i].first!=transitions[i].second){
            // makes transition if ASP returns true for the observation
            if(InterpretBool(solution_preds[i], obsObject)) {
                state.ha = to_label(transitions[i].second);
                break;
            }
        }
    }

    if(synthesizer == EMDIPS || !use_error) {
        return state.ha;
    }
    // Allow a little error in LDIPS
    return correct(prev_ha, pointError(prev_ha, state.ha, POINT_ACCURACY));
}

// Initial ASP: random transitions
HA initialASP(State state) {
    if(labeler == GREEDY_HEURISTIC) {
        return correct(state.ha, pointError(state.ha, state.ha, 0));
    } else {
        return correct(state.ha, pointError(state.ha, state.ha, POINT_ACCURACY));
    }
}

void save_metric(string output_path, double metric) {
    ofstream info_file;
    info_file.open(output_path, ios::app);
    info_file << metric << endl;
    info_file.close();
}

void print_metrics(double cum_log_obs, double ha_correct, double t_total, DATATYPE data) {
    if(t_total == 0) {
        return;
    }

    cum_log_obs /= t_total; // Normalize
    ha_correct /= t_total; ha_correct *= 100; // Convert to percentage

    cout << "Metrics";
    switch(data) {
        case TRAINING:
            cout << " (training):\n"; break;
        case TESTING:
            cout << " (testing):\n"; break;
        case VALIDATION:
            cout << " (validation):\n"; break;
    }
    cout << "\tCumulative observation likelihood: e^" << cum_log_obs << " = " << exp(cum_log_obs) << endl;
    cout << "\t%% Accuracy: " << ha_correct << "%%" << endl;

    switch(data) {
        case TRAINING:
            save_metric(LOG_OBS_PATH + "-training.txt", cum_log_obs); save_metric(PCT_ACCURACY + "-training.txt", ha_correct);
            break;
        case TESTING:
            save_metric(LOG_OBS_PATH + "-testing.txt", cum_log_obs); save_metric(PCT_ACCURACY + "-testing.txt", ha_correct);
            break;
        case VALIDATION:
            save_metric(LOG_OBS_PATH + "-valid.txt", cum_log_obs); save_metric(PCT_ACCURACY + "-valid.txt", ha_correct);
            break;
    }
}

/**
 * @brief 
 * 
 * @param traj Current trajectory
 * @param asp Current ASP policy
 * @param output_path File path to save output data 
 * @param ha_correct out param
 * @param ha_total out param
 * @return double, log_obs_cum: 
 */
double save_pure(Trajectory traj, asp* asp, string output_path, double& ha_correct, double& ha_total) {    
    ofstream outFile;
    outFile.open(output_path + ".csv");

    // creates a list of files, one for each lower action (i.e. variables the agent controls)
    map<string, shared_ptr<ofstream>> la_files;
    for(string each: LA_vars) {
        la_files[each] = make_shared<ofstream>(output_path + "-" + each + ".csv");
    }

    Trajectory gt = traj; // save current trajectory as a ground truth
    double log_obs_cum = 0;

    use_error = false;

    for(uint32_t n = 0; n < SAMPLE_SIZE; n++){
        log_obs_cum += execute_pure(traj, asp);
        // -> traj updated using the newly synthesized policy

        for(int t = 0; t < traj.size() - 1; t++){
            ha_total++;
            if(traj.get(t+1).ha == gt.get(t+1).ha){
                ha_correct++; // calculate new policy's accuracy
            }

            State last = (t == 0) ? State () : traj.get(t-1);
            State cur = traj.get(t);
            for(string each: LA_vars) {
                cur.put(each, last.get(each));
            }
            Obs la = motorModel(cur, false);
            for(string each: LA_vars) {
                (*la_files[each]) << la.get(each) << ",";
            }
        } //// 

        Obs la = motorModel(traj.get(traj.size()-1), false);
        for(string each: LA_vars) {
            (*la_files[each]) << la.get(each) << endl;
        }

        outFile << traj.to_string();
    }
    
    outFile.close();
    for(string each: LA_vars) {
        la_files[each]->close();
    }

    use_error = true;
    return log_obs_cum;
}

// Expectation step
/**
 * @brief Runs particle filter with asp and converts all states to the AST data type Example （不是我怎么感觉跟理论中的expectation没啥关系呢？？）
 * 
 * @param iteration Current iteration of the EM loop
 * @param state_traj List of Trajectories read from demonstrations; one file translates to one trajectory
 * @param asp Current ASP policy
 * @return vector<vector<Example>> 
 */
vector<vector<Example>> expectation(uint iteration, vector<Trajectory>& state_traj, asp* asp){

    vector<vector<Example>> examples;

    cout << "Running particle filter with " << NUM_PARTICLES << " particles\n";
    cout << "Parameters: resample threshold=" << RESAMPLE_THRESHOLD << ", observation strength=" << TEMPERATURE << endl;

    double cum_log_obs = 0;
    double ha_total = 0, ha_correct = 0;
    // for first 10 (defined as TRAINING_SET) demonstrations
    for(uint i = 0; i < TRAINING_SET; i++){
        string in = SIM_DATA + to_string(i) + ".csv";
        string out = TRAINING_TRAJ + to_string(iteration) + "-" + to_string(i) + ".csv"; // outputs to "iter#-#.csv" (replacing # with a number)
        examples.push_back(vector<Example>());

        // 创造trajectories object  -- stores SAMPLE_SIZE number of trajectories after filtering, also written to out/training_traj folder per iteration
        vector<vector<HA>> trajectories;
        
        // Run filter
        // 调用particle filter (filtered again based on the latest asp)
        // filter state_traj[i] 然后存入 trajectories
        cum_log_obs += filterFromFile(trajectories, NUM_PARTICLES, SAMPLE_SIZE, in, out, state_traj[i], asp);

        // 随机排序trajectories
        shuffle(begin(trajectories), end(trajectories), default_random_engine {});
        
        // Convert each particle trajectory point to EMDIPS-supported Example (好像只是把每个state的数据转换成了AST里用的数据结构)
        for(uint n = 0; n < trajectories.size(); n++){ // for every traj in trajectories
            vector<HA> traj = trajectories[n];
            for(int t = 0; t < state_traj[i].size() - 1; t++){
                Example ex = dataToExample(traj[t], state_traj[i].get(t+1).obs); // 把filter后的每个trajectory与下一个time stamp的observation变成Example object (used for AST)

                // Provide next high-level action
                ex.result_ = SymEntry(print(traj[t+1]));
                examples[i].push_back(ex);

                ha_total++;

                // ha_correct: matches between before vs after particle filter
                if(traj[t+1] == state_traj[i].get(t+1).ha) {
                    ha_correct++;
                }
            }
        }

        cout << "*";
        cout.flush();
    }

    // 以下都是print logs
    cout << "\n"
    print_metrics(cum_log_obs, ha_correct, ha_total, TRAINING); // cum_log_obs results from particle filter
    
    ha_correct = ha_total = cum_log_obs = 0; // resets the variables
    for(uint i = 0; i < VALIDATION_SET; i++){
        // after 10 sets (defined in settings), print training metrics and reset variables again
        if(i == TRAINING_SET) {
            print_metrics(cum_log_obs, ha_correct, ha_total, TESTING);
            ha_correct = ha_total = cum_log_obs = 0;
        }

        string plot = VALIDATION_TRAJ+to_string(iteration)+"-"+to_string(i);
        cum_log_obs += save_pure(state_traj[i], asp, plot, ha_correct, ha_total);
    }
    print_metrics(cum_log_obs, ha_correct, ha_total, VALIDATION);

    return examples; // return了Example的列表
}

void default_merge(vector<vector<Example>>& allExamples, vector<Example>& consolidated){
    for(vector<Example>& each : allExamples){
        consolidated.insert(end(consolidated), begin(each), end(each));
    }
}

/**
 * @brief Maximization step
 * 
 * @param allExamples output from the expectation step, converted from trajectories
 * @param iteration 
 */
void maximization(vector<vector<Example>>& allExamples, uint iteration){
    vector<Example> samples;
    default_merge(allExamples, samples); // consolidates allExamples to a vector, instead of a vector of vectors
    
    // Setup output for ASPs and accuracies
    string aspFilePath = GEN_ASP + to_string(iteration) + "/";
    filesystem::create_directory(aspFilePath);

    vector<ast_ptr> inputs; vector<Signature> sigs;
    vector<ast_ptr> ops = AST::RecEnumerate(roots, inputs, samples, library, BASE_FEAT_DEPTH, &sigs);
    if (iteration % STRUCT_CHANGE_FREQ != 0) {
        // Don't enumerate: only optimize current sketch
        ops.clear();
    }

    cout << "---- Number of Features Enumerated ----" << endl;
    cout << ops.size() << endl << endl;
    for(int i = 0; i < min(5, (int) ops.size()); i++){
        cout << ops[i] << endl;
    }
    cout << "...\n\n\n";

    // Run synthesis algorithm to optimize sketches
    if(synthesizer == EMDIPS) { // EMDIPS
        emdipsL3(samples, transitions, solution_preds, loss, ops, aspFilePath, pFunc); // 调用了pips/src/ast/synthesis.cpp里的ASP synthesis函数，根据当前的demonstrations合成了ASP
    } else { // LDIPS
        ldipsL3(samples, transitions, solution_preds, loss, ops, aspFilePath);
    }

    // Write ASP info to file
    ofstream aspStrFile;
    string aspStrFilePath = GEN_ASP + to_string(iteration) + "/asp.txt";
    aspStrFile.open(aspStrFilePath);
    for(uint i = 0; i < transitions.size(); i++){
        aspStrFile << transitions[i].first + " -> " + transitions[i].second << endl;
        aspStrFile << "Loss: " << loss[i] << endl;
        aspStrFile << solution_preds[i] << endl;
    }
    aspStrFile.close();
}

/** Settings
 * @brief Pass in settings from settings.h and include.h, and sets valid transitions, etc
 * 
 */
void setupLdips(){
    cout << "-------------Setup----------------" << endl;

    passSettings(PROG_ENUM, PROG_COMPLEXITY_LOSS_BASE, PROG_COMPLEXITY_LOSS, synth_setting == INCREMENTAL);

    for(Var each: Obs_vars) { // observation variables, defined in domain.h
        if(each.root_) {
            roots.push_back(make_shared<Var>(each));
        }
    }

    // Insert transitions
    for(uint i = 0; i < numHA; i++){
        vector<HA> valid_ha = get_valid_ha(i);
        for(uint j = 0; j < valid_ha.size(); j++){
            if(i != valid_ha[j]){
                transitions.push_back(pair<string, string> (print(i), print(valid_ha[j])));
                loss.push_back(numeric_limits<float>::max());
            }
        }
    }
    
    std::sort(transitions.begin(), transitions.end(), [](const pair<string, string>& a, const pair<string, string>& b) -> bool {
        if(a.first == b.first){
            if(a.first == a.second) return false;
            if(b.first == b.second) return true;
            return a.second < b.second;
        }
        return a.first < b.first;
    });

    cout << "----Roots----" << endl;
    for (auto& node : roots) {
        cout << node << endl;
    }
    cout << endl;

    cout << "----Transitions----" << endl;
    for (auto& trans : transitions) {
        cout << trans.first << "->" << trans.second << endl;
    }
    cout << endl;
}

/**
 * @brief Read demonstrations from csv files and run particle filters on them
 * 
 * @param state_traj out param; Stores list of Trajectories read from demonstrations
 */
void read_demonstration(vector<Trajectory>& state_traj){

    cout << "\nReading demonstration and optionally running ground-truth ASP..." << endl;

    for(int r = 0; r < VALIDATION_SET; r++){ // VALIDATION_SET = 30, 一共30份demonstration数据
        string inputFile = SIM_DATA + to_string(r) + ".csv"; // python-gen/data#.csv
        Trajectory traj;
        readData(inputFile, traj); // 从csv读取并存储到Trajectory数据结构里（调用的particleFilter/pf_runner.h的函数）

        state_traj.push_back(traj); // 加到state_traj的列表里
    }

    double cum_log_obs = 0, ha_total = 0, ha_correct = 0;
    // 只跑前10个data文档当training set（可在settings.h修改TRAINING_SET数量)
    for(uint i = 0; i < TRAINING_SET; i++){
        string in = SIM_DATA + to_string(i) + ".csv";
        string out = TRAINING_TRAJ + "gt-" + to_string(i) + ".csv"; // outputs to "itergt-#.csv" (replacing # with a number); 这里gt应该指的是ground truth

        // Run filter
        vector<vector<HA>> trajectories;
        cum_log_obs += filterFromFile(trajectories, NUM_PARTICLES, SAMPLE_SIZE, in, out, state_traj[i], ASP_model); // ground truth的filter结果，写入了training_traj/itergt-#.csv，用了手写的ASP

        for(uint n = 0; n < trajectories.size(); n++){
            for(int t = 0; t < state_traj[i].size() - 1; t++){
                ha_total++;
                if(trajectories[n][t+1] == state_traj[i].get(t+1).ha) {
                    ha_correct++;
                }
            }
        }
    }

    // print training log
    print_metrics(cum_log_obs, ha_correct, ha_total, TRAINING);
    
    ha_correct = ha_total = cum_log_obs = 0;
    for(uint i = 0; i < VALIDATION_SET; i++){
        if(i == TRAINING_SET) {
            print_metrics(cum_log_obs, ha_correct, ha_total, TESTING); // print testing log
            ha_correct = ha_total = cum_log_obs = 0;
        }
        
        string plot = VALIDATION_TRAJ+"gt-"+to_string(i);
        cum_log_obs += save_pure(state_traj[i], ASP_model, plot, ha_correct, ha_total);
    }
    print_metrics(cum_log_obs, ha_correct, ha_total, VALIDATION); // print validation log
}

// 主算法
void emLoop(){

    // Initialization
    setupLdips();

    library = ReadLibrary(OPERATION_LIB);
    asp* curASP = initialASP;
    vector<Trajectory> state_traj;
    read_demonstration(state_traj);

    for(int i = 0; i < NUM_ITER; i++){
        auto start = chrono::system_clock::now();

        // Expectation
        cout << "\n|-------------------------------------|\n";
        cout << "|                                     |\n";
        cout << "|          Loop " << i << " EXPECTATION         |\n";
        cout << "|                                     |\n";
        cout << "|-------------------------------------|\n";
        vector<vector<Example>> examples = expectation(i, state_traj, curASP);

        // Maximization
        cout << "\n|-------------------------------------|\n";
        cout << "|                                     |\n";
        cout << "|         Loop " << i << " MAXIMIZATION         |\n";
        cout << "|                                     |\n";
        cout << "|-------------------------------------|\n";
        maximization(examples, i);

        curASP = emdipsASP;
        TEMPERATURE = max(TEMPERATURE - TEMP_CHANGE, 1.0);

        auto end = chrono::system_clock::now();
        chrono::duration<double> diff = end - start;

        cout << "--------------- Iteration " << i << " took " << diff.count() << " s ---------------" << endl;
    }
}



pair<PyObject*, PyObject*> setup_python(){
    
    Py_Initialize();

    string s = "import os, sys \nsys.path.append(os.getcwd() + '/"+OPTIMIZER_PATH+"') \n";
    PyRun_SimpleString(s.c_str());

    PyObject* pName = PyUnicode_FromString((char*)"optimizer");
    PyObject* pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // Function name
        pFunc = PyObject_GetAttrString(pModule, (char*)"run_optimizer_threads");

        if (!(pFunc && PyCallable_Check(pFunc))) {
            if (PyErr_Occurred()) PyErr_Print();
            fprintf(stderr, "Cannot find optimization function\n");
        }
    } else {
        PyErr_Print();
        fprintf(stderr, "Failed to load optimization file\n");
    }
    return pair(pFunc, pModule);
}



void cleanup_python(pair<PyObject*, PyObject*> py_info){
    auto pFunc = py_info.first;
    auto pModule = py_info.second;
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    Py_Finalize();
}

int main() {

    auto py_info = setup_python();
    emLoop();
    cleanup_python(py_info);

    return 0;
}

#pragma once

#include "domain.h"
#include "system.h"

using namespace std;
using namespace SETTINGS;

// MOTOR (OBSERVATION) MODEL: known function mapping from high-level to low-level actions
map<string, normal_distribution<double>> la_error = {
    { "steer", normal_distribution<double>(0.0, 0.02) },
    { "acc", normal_distribution<double>(0.0, 2) }
};

double TURN_HEADING = 0.15;
double TURN_TARGET = 30;
double max_velocity = 45;

double lane_diff = 4;

double laneFinder(double y) {
    return round(y / lane_diff);
}

// Motor model matches highway2d.py
Obs motorModel(State state, bool error){
    HA ha = state.ha;

    double target_acc = 0.0;
    double target_heading = 0.0;
    
    if(ha == FASTER) {
        // Attain max speed
        target_acc = max_velocity - state.get("vx");

        // Follow current lane
        double target_y = laneFinder(state.get("y")) * lane_diff;
        target_heading = atan((target_y - state.get("y")) / TURN_TARGET);
    } else if (ha == SLOWER) {
        // Attain min speed
        target_acc = state.get("f_vx") - state.get("vx");

        // Follow current lane
        double target_y = laneFinder(state.get("y")) * lane_diff;
        target_heading = atan((target_y - state.get("y")) / TURN_TARGET);
    } else if (ha == LANE_LEFT) {
        target_acc = -0.5;

        // Attain leftmost heading
        target_heading = -TURN_HEADING;
    } else if (ha == LANE_RIGHT) {
        target_acc = -0.5;

        // Attain rightmost heading
        target_heading = TURN_HEADING;
    }

    double target_steer = target_heading - state.get("heading");
    if(target_steer > state.get("steer")) {
        target_steer = min(target_steer, state.get("steer") + 0.08);
    } else {
        target_steer = max(target_steer, state.get("steer") - 0.08);
    }

    if(target_acc > state.get("acc")) {
        target_acc = min(target_acc, state.get("acc") + 4);
    } else {
        target_acc = max(target_acc, state.get("acc") - 6);
    }

    state.put("steer", target_steer);
    state.put("acc", target_acc);

    return state.obs;
}

HA ASP_model(State state){
    // bool front_clear = flip(logistic(1, 80, (state.get("f_x") - state.get("x")) / state.get("vx")));

    // if(state.ha == LANE_RIGHT) {
    //     bool right_clear = flip(logistic(0.5, 80, (state.get("r_x") - state.get("x")) / state.get("vx")));
    //     if(right_clear)
    //         return LANE_RIGHT;
    //     if(front_clear) 
    //         return FASTER;

    //     return SLOWER;
    // }

    // bool right_clear = flip(logistic(1, 80, (state.get("r_x") - state.get("x")) / state.get("vx")));
    // if(right_clear)
    //     return LANE_RIGHT;
    // if(front_clear) 
    //     return FASTER;

    // return SLOWER;

    if(state.ha == FASTER) {
        // if(flip(logistic(40, 96, state.get("l_vx")))) {
        //     return LANE_LEFT;
        // }
        if(flip(logistic(29.36, 2.64, state.get("r_x") - state.get("x")))) {
            return LANE_RIGHT;
        }
        if(flip(logistic(0.987, -21.34, (state.get("f_x") - state.get("x")) / state.get("vx") ))) {
            return SLOWER;
        }
        return FASTER;
    } else if (state.ha == LANE_LEFT) {
        if(flip(logistic(24.84, 14.665, state.get("r_vx")))) {
            return FASTER;
        }
        if(flip(logistic(417.2768, -3.05, state.get("r_x")))) {
            return LANE_RIGHT;
        }
        return LANE_LEFT;
    } else if (state.ha == LANE_RIGHT) {
        if(flip(logistic(-17.2669, -1.214, state.get("x") - state.get("f_x"))) && flip(logistic(-0.5384, 43201.3125, (state.get("x") - state.get("r_x")) / state.get("vx")))) {
            return FASTER;
        }
        if(flip(logistic(10.06, 153.83, state.get("y")))) {
            return SLOWER;
        }
        return LANE_RIGHT;
    } else {
        if(flip(logistic(-1668.47, -0.003233, state.get("r_x")))) {
            return FASTER;
        }
        // if(flip(logistic(629.19, 0.02975, state.get("f_x")))) {
        //     return LANE_LEFT;
        // }
        if(flip(logistic(-28.716, -1, state.get("x") - state.get("r_x")))) {
            return LANE_RIGHT;
        }
        return SLOWER;
    }
}

Obs physicsModel(State state, double t_step){}

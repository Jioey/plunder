import gym
import highway_env
from matplotlib import pyplot as plt
import os
import numpy as np
import math
import random
from typing import List, Tuple, Union, Optional

######## Configuration ########
lane_diff = 0.25 # Distance lanes are apart from each other
lanes_count = 4 # Number of lanes
use_absolute_lanes = True # Whether or not to label lanes as absolute or relative to current vehicle lane

env = gym.make('highway-v0')
env.config['simulation_frequency']=20
env.config['policy_frequency']=2 # Runs once every 10 simulation steps
env.config['lanes_count']=lanes_count

# Observations
# ego vehicle:      presence, x, y, vx, vy
# 9 other vehicles: presence, x, y, vx, vy        (relative to ego)
env.config['observation']={
    'type': 'Kinematics',
    'vehicles_count': 10,
    'features': ['presence', 'x', 'y', 'vx', 'vy'],
    'absolute': False
}

ACTIONS_ALL = { # A mapping of action indexes to labels
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTION_REORDER = { # highway-env uses a different order for actions (desired: FASTER, SLOWER, LANE_LEFT, LANE_RIGHT)
    0: 2,
    1: -1,
    2: 3,
    3: 0, 
    4: 1
}

highway_env.highway_env.envs.MDPVehicle.DEFAULT_TARGET_SPEEDS = np.linspace(18, 30, 5) # Speed interval (lower_bound, upper_bound, num_samples)
highway_env.highway_env.envs.ControlledVehicle.DELTA_SPEED = 6 # Acceleration / Deceleration

######## ASP ########
# Probabilistic functions
def logistic(offset, slope, x):
    return 1.0/(1.0+np.exp(-slope*(x-offset)))

def sample(p):
    return random.random()<p

# Helper functions

# Round y-positions to the nearest lane
def classifyLane(obs):
    for vehicle in obs:
        vehicle[2] = round(vehicle[2] / lane_diff)
    return obs

# Find closest vehicles in lanes next to the ego vehicle
# Assumption: vehicles are already sorted based on x distance (ignores vehicles behind the ego vehicle)
def closestInLane(obs, lane):
    for vehicle in obs:
        if vehicle[0] == 0: # not present
            continue
        if vehicle[2] == lane: # in desired lane
            return vehicle
    
    return [0, 1000000000, lane, 0, 0] # No car found

def closestVehicles(obs):
    closestLeft = closestInLane(obs[1:], -1)
    closestFront = closestInLane(obs[1:], 0)
    closestRight = closestInLane(obs[1:], 1)

    # Handle edges (in rightmost or leftmost lane)
    if obs[0][2] == 0: # In leftmost lane: pretend there is a vehicle to the left
        closestLeft = obs[0]
        closestLeft[1] = 0
        closestLeft[2] = -1
    if obs[0][2] == lanes_count - 1: # In rightmost lane: pretend there is a vehicle to the right
        closestRight = obs[0]
        closestRight[1] = 0
        closestRight[2] = 1
    
    return (closestLeft, closestFront, closestRight)


# ASP (probabilistic)
def prob_asp(ego, closest):
    front_clear = sample(logistic(0.15, 90, closest[1][1]))
    left_clear = sample(logistic(0.15, 90, closest[0][1]))
    right_clear = sample(logistic(0.15, 90, closest[2][1]))

    # Deterministic version
    # front_clear = closest[1][1] > 0.15
    # left_clear = closest[0][1] > 0.15
    # right_clear = closest[2][1] > 0.15

    if front_clear: # No car in front: accelerate
        return env.action_type.actions_indexes["FASTER"]
    if left_clear: # No car on the left: merge left
        return env.action_type.actions_indexes["LANE_LEFT"]
    if right_clear: # No car on the right: merge right
        return env.action_type.actions_indexes["LANE_RIGHT"]

    # Nowhere to go: decelerate
    return env.action_type.actions_indexes["SLOWER"]

# copied from https://github.com/eleurent/highway-env/blob/31881fbe45fd05dbd3203bb35419ff5fb1b7bc09/highway_env/vehicle/controller.py
# which also contains motor model
def get_la(self, action):
    # We copy these values to avoid running each action twice
    speed = self.target_speed
    lane_index = self.target_lane_index

    if action == env.action_type.actions_indexes["FASTER"]:
        speed += self.DELTA_SPEED
    elif action == env.action_type.actions_indexes["SLOWER"]:
        speed -= self.DELTA_SPEED
    elif action == env.action_type.actions_indexes["LANE_RIGHT"]:
        _from, _to, _id = self.target_lane_index
        target_lane_index = _from, _to, np.clip(_id + 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
            lane_index = target_lane_index
    elif action == env.action_type.actions_indexes["LANE_LEFT"]:
        _from, _to, _id = self.target_lane_index
        target_lane_index = _from, _to, np.clip(_id - 1, 0, len(self.road.network.graph[_from][_to]) - 1)
        if self.road.network.get_lane(target_lane_index).is_reachable_from(self.position):
            lane_index = target_lane_index

    la = {"steering": self.steering_control(lane_index),
            "acceleration": self.speed_control(speed)}
    la['steering'] = np.clip(la['steering'], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
    
    return la

######## Simulation ########
for iter in range(5):

    env.reset()
    ha = env.action_type.actions_indexes["FASTER"]
    obs_out = open("data" + str(iter) + ".csv", "w")
    obs_out.write("left_present, l_x, l_y, l_vx, l_vy, forward_present, f_x, f_y, f_vx, f_vy, right_present, r_x, r_y, r_vx, r_vy, LA.steer, LA.acc, HA, target_lane\n")

    for _ in range(100):

        obs, reward, done, truncated, info = env.step(ha)
        env.render()

        # Pre-process observations
        obs = classifyLane(obs)
        closest = closestVehicles(obs)

        # Run ASP
        ha = prob_asp(obs[0], closest)

        # Run motor model
        la = get_la(env.vehicle, ha)

        for v in closest:
            for prop in v:
                obs_out.write(str(round(prop, 3))+", ")
        obs_out.write(str(round(la['steering'], 3))+", ")
        obs_out.write(str(round(la['acceleration'], 3))+", ")
        obs_out.write(str(ACTION_REORDER[ha])+", ")
        obs_out.write(str(env.vehicle.target_lane_index[2]))
        obs_out.write("\n")


    obs_out.close()

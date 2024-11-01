import gym
from highway_env.envs import MDPVehicle, ControlledVehicle, Vehicle, highway_env
from highway_env.envs.common.observation import KinematicObservation
from highway_env.envs.common import graphics
from highway_env.utils import near_split, class_from_path
import numpy as np
import random
from typing import Union

def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

highway_env.HighwayEnv._create_vehicles = _create_vehicles

######## Configuration ########
lane_diff = 4 # Distance lanes are apart from each other
lanes_count = 4 # Number of lanes
use_absolute_lanes = True # Whether or not to label lanes as absolute or relative to current vehicle lane
KinematicObservation.normalize_obs = lambda self, df: df # Don't normalize values

steer_err = 0.01
acc_err = 2

env = gym.make('highway-v0')
env.config['simulation_frequency']=24
env.config['policy_frequency']=8 # Runs once every 3 simulation steps
env.config['lanes_count']=lanes_count
env.config['initial_lane_id'] = 0
# env.configure({
    # "manual_control": True
# })

# Observations
# ego vehicle:      presence, x, y, vx, vy, heading
# 9 other vehicles: presence, x, y, vx, vy, heading        (x and y relative to ego)
env.config['observation']={
    'type': 'Kinematics',
    'vehicles_count': 10,
    'features': ['presence', 'x', 'y', 'vx', 'vy', 'heading'],
    'absolute': True
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

######## ASP ########
# Probabilistic functions
def logistic(offset, slope, x):
    return 1.0/(1.0+np.exp(-slope*(x-offset)))

def sample(p):
    return random.random()<p

# Helper functions

# Round y-positions to the nearest lane
def laneFinder(y):
    return round(y / lane_diff)

def classifyLane(obs):
    lane_class = []
    for vehicle in obs:
        lane_class.append(laneFinder(vehicle[2]))
    return lane_class

# Find closest vehicles in lanes next to the ego vehicle
# Assumption: vehicles are already sorted based on x distance (ignores vehicles behind the ego vehicle)
def closestInLane(obs, lane, lane_class, ego):
    for i in range(len(obs)):
        if obs[i][0] == 0: # not present
            continue
        if lane_class[i] == lane: # in desired lane
            return obs[i]
    
    return [0, ego[1] + 100, lane * lane_diff, ego[3], ego[4], ego[5]] # No car found

def closestVehicles(obs, lane_class):
    ego_lane = laneFinder(obs[0][2])

    closestLeft = closestInLane(obs[1:], ego_lane - 1, lane_class[1:], obs[0])
    closestFront = closestInLane(obs[1:], ego_lane, lane_class[1:], obs[0])
    closestRight = closestInLane(obs[1:], ego_lane + 1, lane_class[1:], obs[0])

    # Handle edges (in rightmost or leftmost lane)
    if lane_class[0] == 0: # In leftmost lane: pretend there is a vehicle to the left
        closestLeft = obs[0].copy()
        closestLeft[2] = obs[0][2] - lane_diff
    if lane_class[0] == env.config['lanes_count'] - 1: # In rightmost lane: pretend there is a vehicle to the right
        closestRight = obs[0].copy()
        closestRight[2] = obs[0][2] + lane_diff
    
    return (closestLeft, closestFront, closestRight)

# ASP (probabilistic)
def prob_asp(ego, closest, ha):
    x, l_x, f_x, r_x = ego[1], closest[0][1], closest[1][1], closest[2][1]
    vx = ego[3]

    if ha == env.action_type.actions_indexes["LANE_RIGHT"]:
        front_clear = sample(logistic(1, 30, (f_x - x) / vx))
        right_clear = sample(logistic(0.75, 30, (r_x - x) / vx)) # time to collision
        if right_clear: # No car on the right: merge right
            return env.action_type.actions_indexes["LANE_RIGHT"]
        if front_clear: 
            return env.action_type.actions_indexes["FASTER"]

        # Nowhere to go: decelerate
        return env.action_type.actions_indexes["SLOWER"]

    right_clear = sample(logistic(1, 30, (r_x - x) / vx)) # time to collision
    if right_clear: # No car on the right: merge right
        return env.action_type.actions_indexes["LANE_RIGHT"]
    
    if ha == env.action_type.actions_indexes["FASTER"]:
        front_clear = sample(logistic(1, 30, (f_x - x) / vx))
    else:
        front_clear = sample(logistic(2, 30, (f_x - x) / vx))
    
    if front_clear: 
        return env.action_type.actions_indexes["FASTER"]

    # Nowhere to go: decelerate
    return env.action_type.actions_indexes["SLOWER"]

# modified from https://github.com/eleurent/highway-env/blob/31881fbe45fd05dbd3203bb35419ff5fb1b7bc09/highway_env/vehicle/controller.py
# in this version, no extra latent state is stored (target_lane, target_speed)
TURN_HEADING = 0.15 # Target heading when turning
TURN_TARGET = 30 # How much to adjust when targeting a lane (higher = smoother)
max_velocity = 40 # Maximum velocity

last_action = "FASTER"
last_target = 0
last_la = {"steering": 0, "acceleration": 0 }

def run_la(self, action: Union[dict, str] = None, step = True, closest = None) -> None:
    global last_action, last_la, last_target

    if step:
        Vehicle.act(self, last_la)
        return last_la

    target_acc = 0.0
    target_heading = 0.0

    if action == None:
        action = last_action
        
    last_action = action

    if action == "FASTER":
        # Attain max speed
        target_acc = max_velocity - self.speed

        # Follow current lane
        target_y = laneFinder(self.position[1]) * lane_diff
        target_heading = np.arctan((target_y - self.position[1]) / TURN_TARGET)
    elif action == "SLOWER":
        # Attain speed of vehicle in front
        if closest == None:
            front_speed = last_target
        else:
            front_speed = closest[1][3]

        last_target = front_speed
        target_acc = (last_target - self.speed)

        # Follow current lane
        target_y = laneFinder(self.position[1]) * lane_diff
        target_heading = np.arctan((target_y - self.position[1]) / TURN_TARGET)
    elif action == "LANE_RIGHT":
        target_acc = -0.5

        # Attain rightmost heading
        target_heading = TURN_HEADING
    elif action == "LANE_LEFT":
        target_acc = -0.5

        # Attain leftmost heading
        target_heading = -TURN_HEADING

    target_steer = target_heading - self.heading

    if target_steer > last_la["steering"]:
        target_steer = min(target_steer, last_la["steering"] + 0.06)
    else:
        target_steer = max(target_steer, last_la["steering"] - 0.06)

    if target_acc > last_la["acceleration"]:
        target_acc = min(target_acc, last_la["acceleration"] + 4)
    else:
        target_acc = max(target_acc, last_la["acceleration"] - 6)

    la = {"steering": target_steer, "acceleration": target_acc }

    # Add error
    la['steering'] = np.random.normal(la['steering'], steer_err)
    la['acceleration'] = np.random.normal(la['acceleration'], acc_err)

    if not closest == None:
        last_la = la
    
    return la

ControlledVehicle.act = run_la

######## Simulation ########
def runSim(iter):
    env.reset()
    graphics.LAST_HA = "FASTER"
    ha = env.action_type.actions_indexes[graphics.LAST_HA]
    obs_out = open("data" + str(iter) + ".csv", "w")
    obs_out.write("x, y, vx, vy, heading, l_x, l_y, l_vx, l_vy, l_heading, f_x, f_y, f_vx, f_vy, f_heading, r_x, r_y, r_vx, r_vy, r_heading, LA.steer, LA.acc, HA\n")

    for _ in range(75):

        obs, reward, done, truncated, info = env.step(ha)
        env.render()

        # Pre-process observations
        lane_class = classifyLane(obs)
        closest = closestVehicles(obs, lane_class)

        # # Run ASP
        # ha = env.action_type.actions_indexes[graphics.LAST_HA]
        ha = prob_asp(obs[0], closest, ha)

        # Run motor model
        la = run_la(env.vehicle, ACTIONS_ALL[ha], False, closest)

        # Ego vehicle
        for prop in obs[0][1:]:
            obs_out.write(str(round(prop, 3))+", ")
        
        # Nearby vehicles
        for v in closest:
            for prop in v[1:]:
                obs_out.write(str(round(prop, 3))+", ")
        obs_out.write(str(round(la['steering'], 3))+", ")
        obs_out.write(str(round(la['acceleration'], 3))+", ")
        obs_out.write(str(ACTION_REORDER[ha]))
        obs_out.write("\n")


    obs_out.close()

for iter in range(25):
    runSim(iter)
iter = 25
# Run generalized simulations involving more vehicles, lanes, etc.

env.config['lanes_count']=lanes_count+2
env.config['observation']={
    'type': 'Kinematics',
    'vehicles_count': 50,
    'features': ['presence', 'x', 'y', 'vx', 'vy', 'heading'],
    'absolute': True
}
runSim(iter)
iter += 1
runSim(iter)
iter += 1
runSim(iter)
iter += 1
runSim(iter)
iter += 1
runSim(iter)
iter += 1
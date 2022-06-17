import matplotlib.pyplot as plt
import numpy as np 
import csv
import sys
import os

# Initialization
trajectories = []
velocities = []
positions = []

gtTrajectory = []
gtVelocity = []
gtPosition = []

settings = {}

# ----- I/O ---------------------------------------------=

# Reads in csv files
def readTrajectories(inFile):
    reader = csv.reader(open(inFile, "r"))
    for x in inFile:
        traj = []
        for elem in next(reader):
            traj.append(float(elem))
        trajectories.append(traj)
    return

def readGroundTruth(gtFile):
    gtReader = csv.reader(open(gtFile, "r"))
    next(gtReader)

    for info in gtReader:
        gtPosition.append(float(info[1]))
        gtVelocity.append(float(info[2]))
        gtTrajectory.append(info[4].strip())
    return

def readSettings(sgFile):
    with open(sgFile) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            vals = line.split(" ")
            settings[vals[0]] = vals[1]
    return


# ----- Simulation ---------------------------------------------

# Populates velocity and position values based on acceleration data
def fillVelocity(acc, dt):
    vel = []
    curVel = 0
    for val in acc:
        curVel += val * dt
        vel.append(curVel)
    return vel

def fillPosition(vel, dt):
    pos = []
    curPos = 0
    lastVel = 0
    for v in vel:
        curPos += 0.5*(lastVel + v)*dt
        lastVel = v
        pos.append(curPos)
    return pos

def runSimulation():
    for traj in trajectories:
        velocities.append(fillVelocity(traj, 0.1))
    
    for vel in velocities:
        positions.append(fillPosition(vel, 0.1))
    
    return


# ----- Main ---------------------------------------------

def main():
    global trajectories, velocities, positions, gtTrajectory, gtVelocity, gtPosition

    # Read settings
    readSettings("settings.txt")
        
    for iter in range(0, int(settings["numIterations"])):
        for robot in range(0, int(settings["numRobots"])):

            trajectories = []
            velocities = []
            positions = []

            gtTrajectory = []
            gtVelocity = []
            gtPosition = []

            # I/O
            inFile = settings["trajGenPath"]+str(iter)+"-"+str(robot)+".csv"
            gtFile = settings["stateGenPath"]+str(robot)+".csv"
            outPath = settings["plotGenPath"]+str(iter)+"-"+str(robot)+"-"

            readTrajectories(inFile)
            readGroundTruth(gtFile)

            max_t = min(int(settings["timeStepsPlot"]), min(len(gtTrajectory), len(trajectories[0])))
            printed_particles = min(int(settings["numParticles"]), len(trajectories))

            # Calculate full state sequences
            runSimulation()

            # Acceleration
            fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [5, 1]})
            fig.suptitle('low-level actions vs. time')
            # for i in range(0, min(printed_particles, len(trajectories))):
            #     traj = trajectories[i]
            #     ax1.plot(traj[:min(len(traj), max_t)], alpha=0.1, linestyle="none", marker="o", color='b') 
            # ax2.plot(gtTrajectory[:min(len(gtTrajectory), max_t)], alpha=0.5, linestyle="none", marker="o", color='b')
            # plt.xlabel('time') 
            # plt.ylabel('acceleration')

            freq = 2
            actions = [[0] * max_t, [0] * max_t, [0] * max_t]
            gt = [[0] * max_t, [0] * max_t, [0] * max_t]
            times = []
            for t in range(0, max_t):
                times.append(t)
                if t % freq == 0:
                    for i in range(0, len(trajectories)):
                        a = trajectories[i][t]
                        if a > 0:
                            actions[0][t] += 1
                        elif a == 0:
                            actions[1][t] += 1
                        elif a < 0:
                            actions[2][t] += 1
                    a = gtTrajectory[t]
                    if a == "ACC":
                        gt[0][t] += 1
                    elif a == "CON":
                        gt[1][t] += 1
                    elif a == "DEC":
                        gt[2][t] += 1
                else:
                    actions[0][t] = actions[0][t-1]
                    actions[1][t] = actions[1][t-1]
                    actions[2][t] = actions[2][t-1]
                    gt[0][t] = gt[0][t-1]
                    gt[1][t] = gt[1][t-1]
                    gt[2][t] = gt[2][t-1]

            ax1.bar(times, actions[0], width=1, bottom=np.add(actions[2], actions[1]), color="#05a655", label="ACC")
            ax1.bar(times, actions[1], width=1, bottom=actions[2], color="#f8ff99", label="CON")
            ax1.bar(times, actions[2], width=1, color="#ff6040", label="DEC")
            
            ax2.bar(times, gt[0], color="#05a655", width=1)
            ax2.bar(times, gt[1], color="#f8ff99", width=1)
            ax2.bar(times, gt[2], color="#ff6040", width=1)
            
            plt.xlabel('time')
            ax1.set_ylabel('low-level action counts')
            ax2.set_ylabel('ground\ntruth')

            ax1.legend(loc="upper left")

            plt.show()
            plt.savefig(outPath + "accel.png")

            plt.clf()

            # Velocity
            for i in range(0, printed_particles):
                vel = velocities[i]
                plt.plot(vel[:max_t], alpha=0.5, linestyle="dashed") 
            plt.plot(gtVelocity[:max_t], alpha=1.0, label="ground truth")
                
            plt.xlabel('time') 
            plt.ylabel('velocity') 
            plt.title('velocity-time') 
            
            plt.legend(loc="upper left")

            plt.show()
            plt.savefig(outPath + "vel.png")

            plt.clf()

            # Position
            for i in range(0, printed_particles):
                pos = positions[i]
                plt.plot(pos[:max_t], alpha=0.5, linestyle="dashed") 
            plt.plot(gtPosition[:max_t], alpha=1.0, label="ground truth")
                
            plt.xlabel('time') 
            plt.ylabel('position') 
            plt.title('position-time')

            plt.legend(loc="upper left") 
            
            plt.show()
            plt.savefig(outPath + "pos.png")

    return


if __name__ == "__main__":
    main()
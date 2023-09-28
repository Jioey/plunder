import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 14})

target = "stack"

# 1D-target
if target == "1D-target":
    # training = [-1.191608, -1.036632, -1.023472, -0.885944, -0.826336, -0.808208, -0.787856, -0.7826184, -0.7741384, -0.7766104, -0.7781736, -0.7775808000000001, -0.7754215999999999, -0.7762136, -0.7797672]
    # testing = [-1.1424826666666668, -0.9647946666666667, -0.955472, -0.9756053333333333, -0.823376, -0.8077946666666667, -0.790472, -0.787408, -0.7835306666666667, -0.784448, -0.7858106666666667, -0.7830773333333333, -0.7815253333333333, -0.782104, -0.7890613333333333]

    training = [86.9776, 91.5504, 92.1128, 95.364, 96.9216, 97.3968, 97.948, 98.1192, 98.2872, 98.2088, 98.1736, 98.2512, 98.2496, 98.28, 98.1856]
    testing = [89.46760000000002, 93.57195000000002, 93.85925, 93.47925, 97.53765000000001, 98.11725000000001, 98.45125, 98.57445, 98.63280000000002, 98.5964, 98.59285000000003, 98.69325, 98.68924999999999, 98.64914999999999, 98.52474999999998]
    
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ren = range(0, len(x)+1, 3)

# 2D-highway
if target == "2D-highway":
    # training = [-1.12849,-0.287779,0.162089,0.261831,0.277573,0.321149,0.259748,0.390868,0.367335,0.387768,0.392703]
    # testing = [-0.4518745,0.0126962,0.0455108,0.3429345,0.292507,0.345308,0.3075035,0.3234745,0.3355485,0.378921,0.3667095]

    training = [82.9624, 89.443, 91.8389, 93.202, 93.4597, 93.7248, 93.3946, 94.1544, 93.6772, 94.0282, 94.0067, 93.0047, 93.953, 93.7933, 94.0275]
    testing = [87.3946, 91.2385, 91.2728, 94.38325, 93.9289, 94.4832, 94.1695, 94.19325, 94.1863, 94.5973, 94.7588, 93.72785, 94.3715, 94.3477, 94.7835]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ren = range(0, len(x)+1, 3)

# 2D-merge
if target == "2D-merge":
    # training = [-2.45907,-1.90358,-1.65009,-1.52427,-1.46253,-1.25326,-0.942493,-0.854028,-0.809816,-0.783505,-0.780442,-0.759502,-0.797437,-0.780687,-0.762582]
    # testing = [-2.64202,-2.13013,-1.87573,-1.70812,-1.64285,-1.33789,-1.05342,-0.966207,-0.918407,-0.894232,-0.895988,-0.9115,-0.894763,-0.885708,-0.891118]
    
    training = [50.0243, 60.0351, 65.6662, 69.0176, 71.8554, 80.0811, 88.8581, 91.5743, 93.0892, 93.7743, 93.9108, 94.2797, 93.477, 93.8703, 94.3446]
    testing = [48.8338, 57.4405, 62.9878, 67.8676, 69.6811, 79.9196, 87.5432, 90.2669, 91.6554, 92.5588, 92.5588, 91.9689, 91.9851, 92.1426, 92.1723]

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    ren = range(0, len(x)+1, 3)

# Pick-and-place
if target == "pick-place":
    # training = [-2.25105, -0.455193, 0.147231, 0.131797, 0.342262, 0.339634, 0.34021, 0.340289, 0.321884, 0.314556]
    # testing = [-2.90212, -0.491461, 0.181175, 0.171733, 0.581842, 0.583377, 0.583154, 0.585468, 0.585548, 0.57876]

    training = [77.1673, 90.5306, 94.9102, 94.8939, 96.2204, 96.1918, 96.2204, 96.2367, 96.1184, 96.3265]
    testing = [73.1374, 89.4054, 93.8844, 93.9034, 96.868, 96.8816, 96.8844, 96.9116, 96.9224, 96.4503]
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ren = range(0, len(x)+1, 2)

# Stack
if target == "stack":
    # training = [-2.90982,-2.00568,-1.48507,-1.41652,-1.20787,-1.42555,-1.40436]
    # testing = [-2.75336,-1.62262,-1.28502,-1.20091,-0.962306,-0.99673,-1.12744]

    training = [69.9886, 85.2658, 91.9812, 93.0309, 94.796, 93.7785, 93.9631]
    testing = [69.402, 85.7698, 91.0879, 92.3275, 93.5946, 94.1047, 93.6322]
    x = [1, 2, 3, 4, 5, 6, 7]

    ren = range(0, len(x)+1, 2)

fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.plot(x, training, label = "training set", linewidth=2)
ax.plot(x, testing, label = "test set", linewidth=2)

plt.xlabel("Iteration")
plt.xticks(ren)
plt.ylabel("Accuracy")

plt.legend(loc = 'lower right')
plt.tight_layout()
plt.grid(linestyle='dotted')
plt.savefig(target + "-emloop.png", dpi=1200)
plt.show()

# LDIPS tests

# # 1D-target
# testing = [-1.10372e+06, -469523, -465382, -465691, -465691, -465691, -465691, -465691, -465691]
# emdips = [-1.10372e+06, -428431, -361798, -358302, -365852, -308766, -302923, -296427, -295278]
# testing = [each / (100*125*30) for each in testing]
# emdips = [each / (100*125*30) for each in emdips]


# x = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# fig, ax = plt.subplots()
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# ax.plot(x, testing, label = "deterministic")
# ax.plot(x, emdips, label = "probabilistic")

# plt.xlabel("Iteration")
# plt.ylabel("Average Obs Likelihood\n(log scale)")

# plt.legend(loc = 'lower right')
# plt.grid()
# plt.savefig("1D-target-ldips.png", dpi=1200)
# plt.show()
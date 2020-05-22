from Reader import DataReader as sr
import EKFSlam as ekf
import FastSLAM as fast
from matplotlib import pyplot as plot
import numpy as np

sensor_list = sr.sensor_reader()
data_list = sr.data_reader()
i = 0
length = 2 * len(data_list) + 3
mut = ekf.initialize(len(data_list))
et = [[0 for i in range(length)] for j in range(length)]

particles = np.random.rand(3, 100)
weights = np.zeros(100)
et_list = list()
q = [[0.1, 0], [0, 0.1]]

for z in range(100):
    et_list.append([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])

f, ax = plot.subplots(nrows=2)
ax[0].set_xlim(-1, 11)
ax[0].set_ylim(-1, 11)
ax[1].set_xlim(-6, 12)
ax[1].set_ylim(-1, 13)
ax[0].scatter(mut[0][0], mut[1][0], c=mut[2][0])
while i < len(sensor_list):
    observations = list()
    observations.append(sensor_list[i])
    i += 1
    while i < len(sensor_list) and sensor_list[i].read_type != 'ODOMETRY':
        observations.append(sensor_list[i])
        i += 1
    mut, et = ekf.calculate_odo(mut, et, observations, data_list)
    particles, et_list, weights = fast.eval_sensor_model(particles, et_list, weights, observations, data_list)
    ax[1].scatter(particles[0], particles[1], c=particles[2])
    ax[0].scatter(mut[0][0], mut[1][0], c=mut[2][0])
    f.canvas.draw()
    plot.pause(0.0000001)

plot.show()

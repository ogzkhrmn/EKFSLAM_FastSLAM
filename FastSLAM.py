import numpy as np
from numpy.linalg import inv
import math

deltamatrix = np.array([0.005, 0.01, 0.005])


def get_result_matrix(input_matrix, sensor_data):
    result_matrix = [[0] for j in range(3)]
    result_matrix[0][0] = input_matrix[0] + sensor_data.data2 * math.cos(input_matrix[2] + sensor_data.data1)
    result_matrix[1][0] = input_matrix[1] + sensor_data.data2 * math.sin(input_matrix[2] + sensor_data.data1)
    result_matrix[2][0] = input_matrix[2] + sensor_data.data1 + sensor_data.data3
    return np.array(result_matrix).transpose()


def get_sensor_result(result, readed):
    result_matrix = [[0] for j in range(2)]
    mx = readed.x
    my = readed.y
    result_matrix[0][0] = math.sqrt(math.pow(mx - result[0][0], 2) + math.pow(my - result[1][0], 2))
    result_matrix[1][0] = math.atan((my - result[1][0]) / (mx - result[0][0])) - result[2][0]
    return np.array(result_matrix).transpose()


def multiply_multi(*args):
    first = np.array(args[0])
    second = np.array(args[1])
    third = np.array(args[2])
    total = first.dot(second)
    return np.array(total).dot(third)


def get_sensor_result(result, readed):
    result_matrix = [[0] for j in range(2)]
    mx = readed.x
    my = readed.y
    result_matrix[0][0] = math.sqrt(math.pow(mx - result[0], 2) + math.pow(my - result[1], 2))
    result_matrix[1][0] = math.atan((my - result[1]) / (mx - result[0])) - result[2]
    return result_matrix


def h_jacob(result, readed):
    mx = readed.x
    my = readed.y
    xt = result[0]
    yt = result[1]
    result_bot = 1 + math.pow((my - yt) / (mx - xt), 2)
    result_matrix = [[xt - mx, yt - my, 0.0],
                     [(1 / math.pow(mx - xt, 2)) / result_bot, -1 / result_bot, -1.0]]
    return result_matrix


def eval_sensor_model(particles, et_list, weights, observations, data_list):
    for index in range(100):
        particles[:, index] = get_result_matrix(particles[:, index], observations[0])
    I = np.identity(3)
    added_landmarks = dict()
    qt = np.array([[0.1, 0], [0, 0.1]])
    qt3 = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    for index in range(100):
        for k in range(len(observations) - 1):
            read_data = observations[k + 1]
            particle = particles[:, index]
            if read_data.data1 not in added_landmarks:
                H_MATRIX = h_jacob(particles[:, index], data_list[int(read_data.data1 - 1)])
                H_MATRIX.append([0.0, 0.0, 0.0])
                et = multiply_multi(np.linalg.pinv(H_MATRIX), qt3, np.array(np.linalg.pinv(H_MATRIX)).transpose())
                et_list[index] = et
                weights[index] = 1
            else:
                H_MATRIX = h_jacob(particle, data_list[int(read_data.data1 - 1)])
                Q = np.add(multiply_multi(H_MATRIX, et_list[index], np.array(H_MATRIX).transpose()), qt)
                K = multiply_multi(et_list[index], np.array(H_MATRIX).transpose(), inv(Q))
                mutul = K.dot(np.subtract(deltamatrix))
                mutu = np.add(mutu, mutul).tolist()
                etu = np.array(np.subtract(I, np.array(K).dot(H_MATRIX))).dot(etu)
                weights[index] = 1
                qu = np.array(2 * np.pi * Q)
                weights[index] = qu.dot(
                    np.exp(-0.5 * multiply_multi(deltamatrix.transpose() * inv(Q) * deltamatrix)))
    return resample_particles(particles, weights), et_list, weights


def resample_particles(particles, weights):
    step = 1.0 / len(particles)
    u = np.random.uniform(0, step)
    c = weights[0]
    i = 0
    new_particles = []
    for particle in particles:
        while u > c:
            i = i + 1
            c = c + weights[i]
        new_particle = particle
        weights[i] = 1.0 / len(particles)
        new_particles.append(new_particle)
        u = u + step
    return np.array(new_particles)

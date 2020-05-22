import math
import statistics
import numpy as np

i_matrix = None
fx = None
mut = None
et = None
rx = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.01]]
length = 0


def initialize(n):
    global et, length, mut, i_matrix, fx
    length = 2 * n + 3
    mut = [[0] for j in range(length)]
    fx = [[0 for i in range(length)] for j in range(3)]
    i_matrix = np.identity(length)
    for j in range(3):
        fx[j][j] = 1
    return mut


def calculate_odo(input_matrix, emat, observations, data_list):
    global et, mut
    et = emat
    mut = input_matrix
    mutu_matrix = get_mutu(input_matrix, observations[0])
    etu_matrix = get_etu(input_matrix, observations[0])
    result = get_result_matrix(input_matrix, observations[0])
    return calculate_observation(mutu_matrix, etu_matrix, result, observations, data_list)


def get_result_matrix(input_matrix, sensor_data):
    result_matrix = [[0] for j in range(3)]
    result_matrix[0][0] = input_matrix[0][0] + sensor_data.data2 * math.cos(input_matrix[2][0] + sensor_data.data1)
    result_matrix[1][0] = input_matrix[1][0] + sensor_data.data2 * math.sin(input_matrix[2][0] + sensor_data.data1)
    result_matrix[2][0] = input_matrix[2][0] + sensor_data.data1 + sensor_data.data3
    return result_matrix


def get_gjac(input_matrix, sensor_data):
    return [[1, 0, -1 * sensor_data.data2 * math.sin(input_matrix[2][0] + sensor_data.data1)],
            [0, 1, sensor_data.data2 * math.cos(input_matrix[2][0] + sensor_data.data1)],
            [0, 0, 1]]


def get_mutu(input_matrix, sensor_data):
    global fx
    return np.transpose(fx).dot(get_result_matrix(input_matrix, sensor_data))


def get_gt(input_matrix, sensor_data):
    global fx, i_matrix
    return np.add(i_matrix, multiply_multi(np.transpose(fx), get_gjac(input_matrix, sensor_data), fx))


def get_etu(input_matrix, sensor_data):
    global et, rx
    gt = get_gt(input_matrix, sensor_data)
    return np.add(multiply_multi(gt, et, np.transpose(gt)), multiply_multi(np.transpose(fx), rx, fx))


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
    result_matrix[0][0] = math.sqrt(math.pow(mx - result[0][0], 2) + math.pow(my - result[1][0], 2))
    result_matrix[1][0] = math.atan((my - result[1][0]) / (mx - result[0][0])) - result[2][0]
    return result_matrix


def get_muj_matrix(mutu, read_data):
    result_matrix = [[0] for j in range(2)]
    rti = read_data[0][0]
    bti = read_data[1][0]
    result_matrix[0][0] = mutu[0][0] + (rti * math.cos(bti + mutu[2][0]))
    result_matrix[1][0] = mutu[1][0] + (rti * math.sin(bti + mutu[2][0]))
    return result_matrix


def get_fxj_matrix(j):
    global length
    fxj = [[0 for i in range(length)] for j in range(5)]
    for i in range(3):
        fxj[i][i] = 1
    for i in range(2):
        fxj[i + 3][int(3 + (2 * j - 2) + i)] = 1
    return fxj


def get_q_matrix(observations):
    q1 = list()
    q2 = list()
    for k in range(len(observations) - 1):
        read_data = observations[k + 1]
        q1.append(read_data.data2)
        q2.append(read_data.data3)
    q_matrix = [[0.01, 0], [0, 0.01]]
    q_matrix[0][0] = statistics.stdev(q1)
    q_matrix[1][1] = statistics.stdev(q2)
    return q_matrix


def calculate_observation(mutu, etu, result, observations, data_list):
    global i_matrix
    added_landmarks = dict()
    q_matrix = get_q_matrix(observations)
    for k in range(len(observations) - 1):
        read_data = observations[k + 1]
        fxj = get_fxj_matrix(read_data.data1)
        zt = get_sensor_result(result, data_list[int(read_data.data1 - 1)])
        if read_data.data1 not in added_landmarks:
            added_landmarks[read_data.data1] = get_muj_matrix(mutu, zt)
        j_matrix = added_landmarks[read_data.data1]
        delta_matrix = [[j_matrix[0][0] - result[0][0]], [j_matrix[1][0] - result[1][0]]]
        q = np.array(delta_matrix).transpose().dot(np.array(delta_matrix))
        q = q[0][0]
        ztu = [[math.sqrt(q)], [math.atan2(delta_matrix[1][0], delta_matrix[0][0]) - result[2][0]]]
        ht_matrix = [[-1 * math.sqrt(q) * delta_matrix[0][0], -1 * math.sqrt(q) * delta_matrix[1][0], 0,
                      math.sqrt(q) * delta_matrix[0][0], math.sqrt(q) * delta_matrix[1][0]],
                     [delta_matrix[1][0], -1 * delta_matrix[0][0], -1 * q, -1 * delta_matrix[1][0], delta_matrix[0][0]]]
        ht = np.array((1 / q) * np.array(ht_matrix).dot(fxj))
        ktr = np.add(multiply_multi(ht, etu, ht.transpose()), q_matrix)
        ktr = np.linalg.inv(ktr)
        kt = multiply_multi(etu.tolist(), ht.transpose().tolist(), ktr.tolist())
        mutul = kt.dot(np.subtract(zt, ztu))
        mutu = np.add(mutu, mutul).tolist()
        etu = np.array(np.subtract(i_matrix, np.array(kt).dot(ht))).dot(etu)
    return mutu, etu

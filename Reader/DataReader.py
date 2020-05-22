from Reader import SensorData as sd
from Reader import World as wd


def sensor_reader():
    datalist = list()
    file = open("sensor_data.dat", "r")
    for data in file:
        data_array = data.replace("\n", "").split(" ")
        datalist.append(sd.SensorData(data_array[0], data_array[1], data_array[2], data_array[3]))
    return datalist


def data_reader():
    datalist = list()
    file = open("world.dat", "r")
    for data in file:
        data_array = data.replace("\n", "").split(" ")
        datalist.append(wd.World(data_array[0], data_array[1], data_array[2]))
    return datalist

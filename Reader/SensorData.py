class SensorData:
    def __init__(self, read_type, data1, data2, data3):
        self.read_type = read_type
        self.data1 = float(data1)
        self.data2 = float(data2)
        self.data3 = float(data3)

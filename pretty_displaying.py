import numpy as np


def pretty(path):
    data = np.load(path, allow_pickle=True).item()
    for key, value in data.items():
        print(key)
        for i in value:
            print(i)
    print('test')


pretty('results/bert/rte/reading.npy')
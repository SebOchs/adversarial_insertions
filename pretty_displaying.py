import numpy as np
import random

def pretty(path):
    data = np.load(path, allow_pickle=True).item()
    xd = []
    for key, value in data.items():
        for i in value:
            x = key + ': ' +  i
            xd.append(x)
            print(x)
    """
    random.shuffle(xd)
    for i in xd[:10]:
        print(i)
    """

pretty('results/bert/seb/ua/reading.npy')
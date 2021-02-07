import numpy as np
import random


def pretty(path):
    data = np.load(path, allow_pickle=True).item()
    display = []
    for key, value in data.items():
        for i in value:
            display.append(key + i)
    print(random.choice(display))
    """
    random.shuffle(xd)
    for i in xd[:10]:
        print(i)
    """


pretty('results/bert/rte/reading.npy')

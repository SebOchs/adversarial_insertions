import numpy as np
import random


def pretty(path):
    # just a function to display reading examples
    data = np.load(path, allow_pickle=True).item()
    display = []
    for key, value in data.items():
        for i in value:
            display.append(key + i)
    # print(random.choice(display))
    """
    for i in display:
        print(i)
    """
    """
    random.shuffle(xd)
    for i in xd[:10]:
        print(i)
    """


pretty('results/T5/msrpc/reading.npy')

import numpy as np
import pprint


def pretty(path):
    # just a function to display reading examples
    data = np.load(path, allow_pickle=True).item()

    for key, value in data.items():
        for i in value:
            pprint.pprint(i)
            print(80*'=')


pretty('results/bert/msrpc/reading.npy')

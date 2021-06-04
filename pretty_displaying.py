import numpy as np
import json

def pretty(path):
    # just a function to display reading examples
    data = np.load(path, allow_pickle=True).item()

    for key, value in data.items():
        for i in value:
            print(json.dumps(i, sort_keys=False, indent=4))
            print(80*'=')


pretty('results/bert/msrpc/bert_reading.npy')

import numpy as np


def analyze(path):
    correct = np.load(path + '/data.npy', allow_pickle=True).item()
    attack = np.load(path + '/attack_data.npy', allow_pickle=True).item()
    attack_results = np.load(path + '/attack_results.npy', allow_pickle=True).item()
    results = np.load(path + '/final_results.npy', allow_pickle=True).item()
    print("Accuracy before attack for " + str(correct['label']) + ': ' + str(len(correct['data'])/correct['length'])[:6])
    print("Accuracy after attack for " + str(correct['label']) + ': ' + str(results['new_accuracy'])[:6])
    print("Success rate: ", sum(attack_results['success'].values()) / sum(attack_results['query'].values()))
    print('test')


analyze('results/bert/msrpc')


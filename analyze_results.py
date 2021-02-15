import numpy as np


def analyze(path):
    """
    Prints the results of an attack on a model / test set combination
    :param path: string / folder path containing attack results
    :return: None
    """
    correct = np.load(path + '/data.npy', allow_pickle=True).item()
    attack = np.load(path + '/attack_data.npy', allow_pickle=True).item()
    attack_results = np.load(path + '/attack_results.npy', allow_pickle=True).item()
    results = np.load(path + '/final_results.npy', allow_pickle=True).item()
    print(path)
    print("Afflicted data instances: ", len(list(attack_results['success'].keys())))
    print("Accuracy before attack for " + str(correct['label']) + ': ' + str(len(correct['data'])/correct['length'])[:6])
    print("Accuracy after attack for " + str(correct['label']) + ': ' + str(results['new_accuracy'])[:6])
    print("Success rate: ", sum(attack_results['success'].values()) / sum(attack_results['query'].values()))





analyze('results/bert/seb/ua')
analyze('results/bert/seb/uq')
analyze('results/bert/seb/ud')
analyze('results/bert/rte')
analyze('results/bert/wic')
analyze('results/bert/msrpc')
analyze('results/bert/mnli/matched')
analyze('results/bert/mnli/mismatched')
analyze('results/T5/seb/ua')
analyze('results/T5/seb/uq')
analyze('results/T5/seb/ud')
analyze('results/T5/rte')
analyze('results/T5/wic')
analyze('results/T5/msrpc')
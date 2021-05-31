import numpy as np


def analyze(path, correct_prediction='data.npy', attack_data='attack_data.npy',
            results='attack_results.npy', final='final_results.npy'):
    """
    Prints the results of an attack on a model / test set combination
    :param path: string / folder path containing attack results
    :return: None
    """
    correct = np.load(path + correct_prediction, allow_pickle=True).item()
    # attack = np.load(path + attack_data, allow_pickle=True).item()
    attack_results = np.load(path + results, allow_pickle=True).item()
    results = np.load(path + final, allow_pickle=True).item()
    print(path)
    print("Afflicted data instances: ", len(list(attack_results['success'].keys())))
    print("Accuracy before attack for " + str(correct['label']) + ': ' + str(len(correct['data'])/correct['length'])[:6])
    print("Accuracy after attack for " + str(correct['label']) + ': ' + str(results['new_accuracy'])[:6])
    print("Success rate: ", sum(attack_results['success'].values()) / sum(attack_results['query'].values()))
    print("Found adversarial examples: ", sum(attack_results['success'].values()))




"""
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
"""
analyze('results/T5/mnli/', correct_prediction='custom_correct_predictions.npy', attack_data='mismatched_attack_data.npy',
        results='mismatched_attack_results.npy')
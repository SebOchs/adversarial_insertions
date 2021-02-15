import nltk
import numpy as np
from transformers import BertTokenizer, T5Tokenizer
from collections import Counter, defaultdict


def analysis(dataset, results, mode):
    """
    Analyzes the occurences of adverbs and adjectives in the training data depending on class
    :param dataset: string / name of data set folder to preprocessed data set
    :param results: string / name of result folder of the attack, contains top performing adv/adj
    :param mode: string / 'bert' or 'T5', based on what adverbs/adjectives should be analyzed
    :return: nothing
    """
    # load tokenizer and prepare data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    data_path = 'datasets/preprocessed/bert/' + dataset + '/train.npy'
    read_path = 'results/' + mode + '/' + results + '/reading.npy'
    data = np.load(data_path, allow_pickle=True)[:, [0, 3]]
    data[:, 0] = [tokenizer.decode(x[0], skip_special_tokens=True) for x in data]
    read = np.load(read_path, allow_pickle=True).item()
    words = list(read.keys())
    labels = list(set(data[:, 1]))

    occurences = {}
    frac_per_label = {}
    # Count occurrences of top adv/adj in the different classes
    for i in labels:
        frac_per_label[str(i)] = data[np.where(data[:, 1] == i)][:, 0]
    for i in words:
        occ = []
        for j in labels:
            occ.append(sum([x.count(i.split('_', 1)[0]) for x in frac_per_label[str(j)]]))
        occurences[i] = occ

    occurences = {k: v for k, v in sorted(occurences.items(), key=lambda item: item[1][-1] - item[1][0], reverse=True)}
    np.save('results/' + mode + '/' + results + '/occurences.npy', occurences, allow_pickle=True)
    # uncomment to see the different occurrences of the top 10
    # print("Dataset: " + dataset, list(occurences.keys())[:10])


analysis('mnli', 'mnli/matched', 'bert')
analysis('seb', 'seb/ua', 'bert')
analysis('msrpc', 'msrpc', 'bert')
analysis('rte', 'rte', 'bert')
analysis('wic', 'wic', 'bert')

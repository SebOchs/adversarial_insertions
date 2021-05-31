import nltk
import numpy as np
import jsonlines
import csv
import os
import xml.etree.ElementTree as et

def analysis(train_set, reading, data_set_name, second_only=False):
    """
    Analyzes the occurences of adverbs and adjectives in the training data depending on class
    :param dataset: string / name of data set folder to preprocessed data set
    :param results: string / name of result folder of the attack, contains top performing adv/adj
    :param mode: string / 'bert' or 'T5', based on what adverbs/adjectives should be analyzed
    :return: nothing
    """

    def create_csv(data, top_ten, csv_path):
        with open(csv_path, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            labels = list(data.keys())
            writer.writerow(['word'] + labels)
            for idx in top_ten:
                row = []
                for label in labels:
                    row.append(data[label][idx])
                writer.writerow([idx] + row)

    # Load most common adjectives and adverbs
    read = np.load(reading, allow_pickle=True).item()
    words = list(read.keys())
    top_ten_adjectives = [x.split('_')[0] for x in words if x.split('_')[1] == 'adj']
    top_ten_adverbs = [x.split('_')[0] for x in words if x.split('_')[1] == 'adv']

    # determine needed column indices
    if data_set_name == 'mnli':
        label_idx = 11
        text_idx = [8, 9]
    if data_set_name == 'msrpc':
        label_idx = 0
        text_idx = [3, 4]
    if data_set_name == 'rte':
        label_idx = 3
        text_idx = [1, 2]

    labels = []
    sentences = []
    if data_set_name == 'wic':
        lines = jsonlines.open(train_set)
        for line in lines:
            labels.append(line['label'])
            if second_only:
                sentences.append(line['sentence2'])
            else:
                sentences.append(line['sentence1'] + ' ' + line['sentence2'])
    elif data_set_name == 'seb':
        for file in os.listdir(train_set):
            root = et.parse(train_set + '/' + file).getroot()
            for ref_answer in root[1]:
                for stud_answer in root[2]:
                    labels.append(stud_answer.get('accuracy'))
                    if second_only:
                        sentences.append(stud_answer.text)
                    else:
                        sentences.append(ref_answer.text + ' ' + stud_answer.text)
    else:
        # Read original training data
        with open(train_set, "r", encoding='utf-8') as file:
            # first line is skipped
            next(file)
            for line in file:
                date = line.strip().split('\t')
                labels.append(date[label_idx])
                text = ''
                if second_only:
                    sentences.append(date[text_idx[1]])
                else:
                    for x in text_idx:
                        text += date[x] + ' '
                    sentences.append(text)

    data = list(zip(sentences, labels))

    collector_adj = {}
    collector_adv = {}
    for i in list(set(labels)):
        tokens = nltk.word_tokenize(' '.join([x[0].lower() for x in data if x[1] == i]))
        adverbs = nltk.FreqDist(x for x in tokens if x in top_ten_adverbs)
        adjectives = nltk.FreqDist(x for x in tokens if x in top_ten_adjectives)
        collector_adj[i] = adjectives
        collector_adv[i] = adverbs

    # create csv's
    if second_only:
        create_csv(collector_adv, top_ten_adverbs, reading.rsplit('/', 1)[0] + '/second_only_adverbs.csv')
        create_csv(collector_adj, top_ten_adjectives, reading.rsplit('/', 1)[0] + '/second_only_adjectives.csv')
    else:
        create_csv(collector_adv, top_ten_adverbs, reading.rsplit('/', 1)[0] + '/adverbs.csv')
        create_csv(collector_adj, top_ten_adjectives, reading.rsplit('/', 1)[0] + '/adjectives.csv')


analysis('datasets/raw/MNLI_matched/train.tsv', 'results/bert/mnli/matched/reading.npy', 'mnli', second_only=True)
analysis('datasets/raw/MNLI_matched/train.tsv', 'results/bert/mnli/mismatched/reading.npy', 'mnli', second_only=True)
analysis('datasets/raw/MSpara/msr_paraphrase_train.txt', 'results/bert/msrpc/reading.npy', 'msrpc', second_only=True)
analysis('datasets/raw/RTE/train.tsv', 'results/bert/rte/reading.npy', 'rte', second_only=True)
analysis('datasets/raw/WiC/train.jsonl', 'results/bert/wic/reading.npy', 'wic', second_only=True)
analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/ua/reading.npy', 'seb', second_only=True)
analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/ud/reading.npy', 'seb', second_only=True)
analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/uq/reading.npy', 'seb', second_only=True)
import spacy
import numpy as np
from collections import defaultdict
import jsonlines
import csv
import os
import xml.etree.ElementTree as et
from tqdm import tqdm
import pandas as pd


def analysis(train_set, bert_attack_results, t5_attack_results, data_set_name):
    """
    Hypothesis testing and adv/adj frequency analysis
    :param train_set: string / path to train set
    :param bert_attack_data: string / path to attack results for bert from selected dataset
    :param t5_attack_data: string / path to attack results for t5 from selected dataset
    :param data_set_name: string / name of dataset
    :return: None
    """

    def load_and_preprocess_dataset(train, second_col, label_col, pos_label, neg_label, json=False):
        if json:
            df = pd.read_json(train_set, lines=True, encoding='utf-8')
        else:
            # load train set
            df = pd.read_csv(train, sep='\t', encoding='utf-8', on_bad_lines='skip')
        # separate into the two required labels
        df_correct = df[df[label_col] == pos_label]
        df_incorrect = df[df[label_col] == neg_label]
        # preprocess the sentence that can be manipulated
        preprocessed_correct = [nlp(x) for x in
                                tqdm(df_correct[second_col].astype(
                                    'str'))]
        preprocessed_incorrect = [nlp(x) for x in
                                  tqdm(df_incorrect[second_col].astype(
                                      'str'))]
        # look up adverbs and adjectives in the texts
        adj_correct = [a for b in [[x.lower_ for x in y if x.pos_ == 'ADJ'] for y in preprocessed_correct] for a in b]
        adv_correct = [a for b in [[x.lower_ for x in y if x.pos_ == 'ADV'] for y in preprocessed_correct] for a in b]
        adj_incorrect = [a for b in [[x.lower_ for x in y if x.pos_ == 'ADJ'] for y in preprocessed_incorrect]
                         for a in b]
        adv_incorrect = [a for b in [[x.lower_ for x in y if x.pos_ == 'ADV'] for y in preprocessed_incorrect]
                         for a in b]
        return adj_correct, adv_correct, adj_incorrect, adv_incorrect

    def build_and_save_data_frames(results, adj_cor, adv_cor, adj_incor, adv_incor):
        # Find 50 most successful adjectives/adverbs
        # first get successful set of insertions per afflicted original
        succ_adv = [a for b in
                         [list(
                             set([x['inserted'] for x in results['adversary_with_info'] if x['original'] == key and
                                  x['type'] == 'ADV'])) for key in list(results['success'].keys())]
                         for a in b]
        succ_adj = [a for b in
                         [list(
                             set([x['inserted'] for x in results['adversary_with_info'] if x['original'] == key and
                                  x['type'] == 'ADJ'])) for key in list(results['success'].keys())]
                         for a in b]
        # get their counts and sort them to get top 50 adjectives / adverbs
        top_adj, adj_count = np.unique(succ_adj, return_counts=True)
        top_adv, adv_count = np.unique(succ_adv, return_counts=True)
        top_50_adj = top_adj[np.argsort(-adj_count)][:50]
        top_50_adv = top_adv[np.argsort(-adv_count)][:50]
        # transform adj/adv occurences in the train set to default dict
        correct_adv_dict = defaultdict(int,
                                       {x[0]: int(x[1]) for x in np.array(np.unique(adv_cor, return_counts=True)).T})
        incorrect_adv_dict = defaultdict(int, {x[0]: int(x[1]) for x in
                                               np.array(np.unique(adv_incor, return_counts=True)).T})
        correct_adj_dict = defaultdict(int,
                                       {x[0]: int(x[1]) for x in np.array(np.unique(adj_cor, return_counts=True)).T})
        incorrect_adj_dict = defaultdict(int, {x[0]: int(x[1]) for x in
                                               np.array(np.unique(adj_incor, return_counts=True)).T})
        adv_df = pd.DataFrame(data=[(x, correct_adv_dict[x], incorrect_adv_dict[x]) for x in top_50_adj],
                              columns=['adv', 'correct_count', 'incorrect_count'])
        adj_df = pd.DataFrame(data=[(x, correct_adj_dict[x], incorrect_adj_dict[x]) for x in top_50_adj],
                              columns=['adj', 'correct_count', 'incorrect_count'])
        return adv_df, adj_df

    nlp = spacy.load("en_core_web_lg", disable=['ner', 'parser', 'lemmatizer', 'tok2vec'])

    # Load attack results
    bert_results = np.load(bert_attack_results, allow_pickle=True).item()
    t5_results = np.load(t5_attack_results, allow_pickle=True).item()

    if data_set_name == 'mnli':
        sent2_col, lab_col, pos_lab, neg_lab, json = "sentence2", "gold_label", "entailment", "neutral", False
    if data_set_name == 'mrpc':
        sent2_col, lab_col, pos_lab, neg_lab, json = "#2 String", "Quality", 1, 0, False
    if data_set_name == 'rte':
        sent2_col, lab_col, pos_lab, neg_lab, json = "hypothesis", "label", "entailment", "not_entailment", True
    if data_set_name == 'wic':
        sent2_col, lab_col, pos_lab, neg_lab, json = "sentence2", "label", True, False, True
    else:
        raise NotImplementedError("Not implemented yet.")

    adj_cor, adv_cor, adj_incor, adv_incor = load_and_preprocess_dataset(train_set, sent2_col, lab_col, pos_lab,
                                                                         neg_lab, json=json)
    # create csv's
    bert_adv, bert_adj = build_and_save_data_frames(bert_results, adj_cor, adv_cor, adj_incor, adv_incor)
    t5_adv, t5_adj = build_and_save_data_frames(t5_results, adj_cor, adv_cor, adj_incor, adv_incor)

    # save them
    bert_adv.to_csv(os.path.join(bert_attack_results.rsplit('/', 1)[0], 'bert_adv.csv'))
    bert_adj.to_csv(os.path.join(bert_attack_results.rsplit('/', 1)[0], 'bert_adj.csv'))
    t5_adv.to_csv(os.path.join(t5_attack_results.rsplit('/', 1)[0], 't5_adv.csv'))
    t5_adj.to_csv(os.path.join(t5_attack_results.rsplit('/', 1)[0], 't5_adj.csv'))

if __name__ == '__main__':
    # MNLI
    """
    analysis('datasets/raw/multinli_1.0/multinli_1.0_train.txt', 'results/bert/mnli/matched/attack_results.npy',
             'results/T5/mnli/matched/matched_attack_results.npy', 'mnli')
    analysis('datasets/raw/multinli_1.0/multinli_1.0_train.txt', 'results/bert/mnli/mismatched/attack_results.npy',
             'results/T5/mnli/mismatched/mismatched_attack_results.npy', 'mnli')
    """
    # MRPC
    """
    analysis('datasets/raw/msrpc/msr_paraphrase_train.txt', 'results/bert/msrpc/attack_results.npy',
             'results/T5/msrpc/attack_results.npy', 'mrpc')
    """
    # RTE
    """
    analysis('datasets/raw/rte/train.jsonl', 'results/bert/rte/attack_results.npy', 'results/T5/rte/attack_results.npy',
             'rte')
    """
    # WiC
    analysis('datasets/raw/wic/train.jsonl', 'results/bert/wic/attack_results.npy', 'results/T5/wic/attack_results.npy',
             'wic')

    # SEB



    """
    analysis('datasets/raw/msrpc/msr_paraphrase_train.txt', 'results/bert/msrpc/reading.npy', 'mrpc')
    analysis('datasets/raw/RTE/train.tsv', 'results/bert/rte/reading.npy', 'rte', second_only=True)
    analysis('datasets/raw/WiC/train.jsonl', 'results/bert/wic/reading.npy', 'wic', second_only=True)
    analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/ua/reading.npy', 'seb', second_only=True)
    analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/ud/reading.npy', 'seb', second_only=True)
    analysis('datasets/raw/sciEntsBank_training', 'results/bert/seb/uq/reading.npy', 'seb', second_only=True)
    """

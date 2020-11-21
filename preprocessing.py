import os
import xml.etree.ElementTree as et
from transformers import BertTokenizer, AlbertTokenizer
import numpy as np


def preprocessing(folder_path, file_path, model):
    def label_to_int(label):
        x = 0
        if label == 'correct':
            x = 2
        if label == 'contradictory':
            x = 1
        return x
    if model[:4] == "bert":
        tokenizer = BertTokenizer.from_pretrained(model)
    elif model[:2] == "al":
        tokenizer = AlbertTokenizer.from_pretrained(model)
    PATH = folder_path
    array = []
    files = os.listdir(PATH)
    for file in files:
        root = et.parse(PATH + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                text_ref = ref_answer.text[:-1]
                text_stud = stud_answer.text[:-1]
                label = label_to_int(stud_answer.get('accuracy'))
                tokenized = tokenizer(text_ref, text_stud, max_length=128, padding='max_length')
                array.append([tokenized.input_ids[:128],
                              tokenized.token_type_ids[:128],
                              tokenized.attention_mask[:128],
                              label])
    np.save(file_path + ".npy", np.array(array), allow_pickle=True)


preprocessing("datasets/raw/sciEntsBank_training", "datasets/preprocessed/bert_sciEntsBank_train", "bert-base-uncased")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-answers", "datasets/preprocessed/bert_sciEntsBank_test_ua", "bert-base-uncased")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-domains", "datasets/preprocessed/bert_sciEntsBank_test_ud", "bert-base-uncased")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-questions", "datasets/preprocessed/bert_sciEntsBank_test_uq", "bert-base-uncased")
preprocessing("datasets/raw/sciEntsBank_training", "datasets/preprocessed/albert_sciEntsBank_train", "albert-large-v2")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-answers", "datasets/preprocessed/albert_sciEntsBank_test_ua", "albert-large-v2")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-domains", "datasets/preprocessed/albert_sciEntsBank_test_ud", "albert-large-v2")
preprocessing("datasets/raw/sciEntsBank_testing/test-unseen-questions", "datasets/preprocessed/albert_sciEntsBank_test_uq", "albert-large-v2")
import os
import xml.etree.ElementTree as et
from transformers import BertTokenizer, T5Tokenizer
import numpy as np
import jsonlines


def save(file_path, data):
    np.save(file_path + ".npy", np.array(data), allow_pickle=True)


def right_tokenizer(model):
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    if model == 'T5':
        tok = T5Tokenizer.from_pretrained('t5-base')
    return tok


def preprocess_seb(file_path, where_to_save, model):
    def label_to_int(label):
        x = 0
        if label == 'correct':
            x = 2
        if label == 'contradictory':
            x = 1
        return x

    tokenizer = right_tokenizer(model)
    array = []
    files = os.listdir(file_path)
    for file in files:
        root = et.parse(file_path + '/' + file).getroot()
        for ref_answer in root[1]:
            for stud_answer in root[2]:
                text_ref = ref_answer.text[:-1]
                text_stud = stud_answer.text[:-1]
                if model == 'bert':
                    label = label_to_int(stud_answer.get('accuracy'))
                    tokenized = tokenizer(text_ref, text_stud, max_length=128, padding='max_length')
                    array.append([tokenized.input_ids[:128],
                                  tokenized.token_type_ids[:128],
                                  tokenized.attention_mask[:128],
                                  label])
                if model == 'T5':
                    label = tokenizer(stud_answer.get('accuracy'), max_length=4, padding='max_length')

                    tokenized = tokenizer("asag: reference:" + text_ref + tokenizer.eos_token + " student: " +
                                          text_stud, max_length=128, padding='max_length')
                    array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                                  label.attention_mask])

    save(where_to_save, array)


def preprocess_mnli(file_path, where_to_save, model):
    def label_to_int(lab):
        if lab == 'neutral':
            return 0
        if lab == 'contradiction':
            return 1
        if lab == 'entailment':
            return 2
        else:
            raise ValueError

    tokenizer = right_tokenizer(model)
    file = jsonlines.open(file_path)
    array = []
    for line in file:
        if line['gold_label'] not in ['entailment', 'contradiction', 'neutral']:
            continue
        if model == 'bert':
            label = label_to_int(line['gold_label'])
            tokenized = tokenizer(line['sentence1'], line['sentence2'], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.token_type_ids[:128], tokenized.attention_mask[:128],
                          label])
        if model == 'T5':
            label = tokenizer(line['gold_label'], max_length=5, padding='max_length')
            tokenized = tokenizer("mnli: premise:" + line['sentence1'] + tokenizer.eos_token + "hypothesis:" +
                                  line['sentence2'], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                          label.attention_mask])

    save(where_to_save, array)


def preprocess_MSpara(file_path, where_to_save, model):
    tokenizer = right_tokenizer(model)
    array = []
    table = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding='utf-8', comments='###')
    for i in table[1:]:
        if model == 'bert':
            label = int(i[0])
            tokenized = tokenizer(i[3], i[4], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.token_type_ids[:128], tokenized.attention_mask[:128],
                          label])
        if model == 'T5':
            label = tokenizer(str(bool(int(i[0]))), max_length=4, padding='max_length')
            tokenized = tokenizer("msrpc: " + "sentence: " + i[3] + tokenizer.eos_token + "paraphrase: " + i[4], max_length=128,
                                  padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                          label.attention_mask])


    save(where_to_save, array)


def preprocess_QQP(file_path, where_to_save, model):
    tokenizer = right_tokenizer(model)
    array = []
    table = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding='utf-8', comments='#1#1')
    for i in table[1:]:
        if model == 'bert':
            label = int(i[5])
            tokenized = tokenizer(i[3], i[4], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.token_type_ids[:128], tokenized.attention_mask[:128],
                          label])
        if model == 'T5':
            label = tokenizer(str(bool(int(i[5]))), max_length=4, padding='max_length')
            tokenized = tokenizer("qqp: question:" + i[3] + tokenizer.eos_token + "duplicate: " + i[4],
                                  max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                          label.attention_mask])


    save(where_to_save, array)


def preprocess_RTE(file_path, where_to_save, model):
    def label_to_int(lab):
        if lab == 'entailment':
            return int(1)
        if lab == 'not_entailment':
            return int(0)
        else:
            raise ValueError

    tokenizer = right_tokenizer(model)
    array = []
    table = np.genfromtxt(file_path, delimiter='\t', dtype=str, encoding='utf-8', comments='#1#1')
    for i in table[1:]:
        if model == 'bert':
            label = label_to_int(i[3])
            tokenized = tokenizer(i[1], i[2], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.token_type_ids[:128], tokenized.attention_mask[:128],
                          label])
        if model == 'T5':
            label = tokenizer(str(bool(label_to_int(i[3]))), max_length=128, padding='max_length')
            tokenized = tokenizer("rte: hypothesis: " + i[1] + tokenizer.eos_token + "premise: " + i[2], max_length=128,
                                  padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                          label.attention_mask])

    save(where_to_save, array)


def preprocess_wic(file_path, where_to_save, model):
    tokenizer = right_tokenizer(model)
    array = []
    table = jsonlines.open(file_path)
    for line in table:
        if model == 'bert':
            label = int(line['label'])
            tokenized = tokenizer(line['sentence1'], line['sentence2'], max_length=128, padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.token_type_ids[:128], tokenized.attention_mask[:128],
                          label])
        if model == 'T5':
            label = tokenizer(str(line['label']), max_length=4, padding='max_length')
            tokenized = tokenizer('wic: word: ' + line['word'] + tokenizer.eos_token + "first:" + line['sentence1'] +
                                  tokenizer.eos_token + "second:" + line['sentence2'], max_length=128,
                                  padding='max_length')
            array.append([tokenized.input_ids[:128], tokenized.attention_mask[:128], label.input_ids,
                          label.attention_mask])


    save(where_to_save, array)

"""
# BERT
# preprocess seb for bert
preprocess_seb('datasets/raw/sciEntsBank_training', 'datasets/preprocessed/bert/seb/train', 'bert')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-answers', 'datasets/preprocessed/bert/seb/test_ua', 'bert')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-domains', 'datasets/preprocessed/bert/seb/test_ud', 'bert')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-questions', 'datasets/preprocessed/bert/seb/test_uq',
                'bert')

# preprocess mnli for bert
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_train.jsonl', 'datasets/preprocessed/bert/MNLI/train',
                'bert')
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_dev_matched.jsonl',
                'datasets/preprocessed/bert/MNLI/dev_m', 'bert')
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_dev_mismatched.jsonl',
                'datasets/preprocessed/bert/MNLI/dev_mm', 'bert')

# preprocess msrpc for bert
preprocess_MSpara('datasets/raw/MSpara/msr_paraphrase_train.txt', 'datasets/preprocessed/bert/MSpara/train', 'bert')
preprocess_MSpara('datasets/raw/MSpara/msr_paraphrase_test.txt', 'datasets/preprocessed/bert/MSpara/test', 'bert')

# preprocess qqp for bert
preprocess_QQP('datasets/raw/QQP/train.tsv', 'datasets/preprocessed/bert/qqp/train', 'bert')
preprocess_QQP('datasets/raw/QQP/dev.tsv', 'datasets/preprocessed/bert/qqp/dev', 'bert')

# preprocess RTE for bert
preprocess_RTE('datasets/raw/RTE/train.tsv', 'datasets/preprocessed/bert/RTE/train', 'bert')
preprocess_RTE('datasets/raw/RTE/dev.tsv', 'datasets/preprocessed/bert/RTE/dev', 'bert')

# preprocess WiC for bert
preprocess_wic('datasets/raw/WiC/train.jsonl', 'datasets/preprocessed/bert/wic/train', 'bert')
preprocess_wic('datasets/raw/WiC/val.jsonl', 'datasets/preprocessed/bert/wic/dev', 'bert')


# T5
"""
# preprocess seb for T5
preprocess_seb('datasets/raw/sciEntsBank_training', 'datasets/preprocessed/T5/seb/train', 'T5')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-answers', 'datasets/preprocessed/T5/seb/test_ua', 'T5')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-domains', 'datasets/preprocessed/T5/seb/test_ud', 'T5')
preprocess_seb('datasets/raw/sciEntsBank_testing/test-unseen-questions', 'datasets/preprocessed/T5/seb/test_uq', 'T5')
"""
# preprocess mnli for T5
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_train.jsonl', 'datasets/preprocessed/T5/MNLI/train',
                'T5')
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_dev_matched.jsonl',
                'datasets/preprocessed/T5/MNLI/dev_m', 'T5')
preprocess_mnli('datasets/raw/MNLI_matched/original/multinli_1.0_dev_mismatched.jsonl',
                'datasets/preprocessed/T5/MNLI/dev_mm', 'T5')

# preprocess msrpc for T5
preprocess_MSpara('datasets/raw/MSpara/msr_paraphrase_train.txt', 'datasets/preprocessed/T5/MSpara/train', 'T5')
preprocess_MSpara('datasets/raw/MSpara/msr_paraphrase_test.txt', 'datasets/preprocessed/T5/MSpara/test', 'T5')

# preprocess qqp for T5
preprocess_QQP('datasets/raw/QQP/train.tsv', 'datasets/preprocessed/T5/qqp/train', 'T5')
preprocess_QQP('datasets/raw/QQP/dev.tsv', 'datasets/preprocessed/T5/qqp/dev', 'T5')

# preprocess RTE for T5
preprocess_RTE('datasets/raw/RTE/train.tsv', 'datasets/preprocessed/T5/RTE/train', 'T5')
preprocess_RTE('datasets/raw/RTE/dev.tsv', 'datasets/preprocessed/T5/RTE/dev', 'T5')


# preprocess WiC for T5
preprocess_wic('datasets/raw/WiC/train.jsonl', 'datasets/preprocessed/T5/wic/train', 'T5')
preprocess_wic('datasets/raw/WiC/val.jsonl', 'datasets/preprocessed/T5/wic/dev', 'T5')
"""

import torch
import numpy as np
from lit_Model import LitBERT, LitT5
import spacy
import time
import utils
import tqdm
from transformers import T5Tokenizer, BertTokenizer
device = torch.device("cuda")

nlp = spacy.load('en_core_web_sm')




def prepare_attack(path, data_path, mode, name):



    def insert_word(word, x, j, tokenizer, ref_answer, btw=''):
        """
        Inserts an adverb/adjective at specific place in a sentence
        :param ref_answer: reference answer as string
        :param word: string, adverb or adjective
        :param x: sentence/sequence preprocessed by spacy
        :param j: integer/index, where param word gets inserted
        :param tokenizer: huggingface tokenizer
        :return: preprocessed data of adversarial sequence
        """
        if j > 0:
            adversarial_ans = x[:j].text + ' ' + word + ' ' + x[j:].text
        else:
            adversarial_ans = word + x.text
        if mode == 'bert':
            new_tokens = tokenizer(ref_answer, adversarial_ans, max_length=128, padding='max_length')
        if mode == 'T5':
            new_tokens = tokenizer(ref_answer + tokenizer.eos_token + btw + adversarial_ans + tokenizer.eos_token,
                                   max_length=128, padding='max_length')
        return new_tokens


    # Load adj and adv
    words = np.load("top_adjectives_adverbs.npy", allow_pickle=True)
    adverbs = words.item()['ADV']
    adjectives = words.item()['ADJ']

    if mode == 'bert':

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        data = np.load(data_path, allow_pickle=True).item()
        attack_data = {"label": data['label'], 'data': []}


        for i in tqdm.tqdm(range(len(data['data'])), desc='Preparing attack data'):

            text, confidence = data['data'][i]
            if len(text) == 2:
                ref, mod = text
            else:
                print('what')
                continue
            pos_tagged = nlp(mod)
            adv_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == 'VERB']
            adj_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == ('NOUN' or 'PROPN' or 'PRON')]
            for a in adv_idx:
                for adverb in adverbs:
                    tokens = insert_word(adverb, pos_tagged, a, tokenizer, ref)
                    attack_data['data'].append({
                        "input": tokens,
                        "original": ''.join(text),
                        "inserted": adverb,
                        "insert_type": 'ADV',
                        "confidence": confidence})

            for b in adj_idx:
                for adjective in adjectives:
                    tokens = insert_word(adjective, pos_tagged, b, tokenizer, ref)
                    attack_data['data'].append({
                        "input": tokens,
                        "original": ''.join(text),
                        "inserted": adjective,
                        "insert_type": 'ADJ',
                        "confidence": confidence})


        np.save(data_path.rsplit('/', 1)[0] + '/attack_data.npy', attack_data, allow_pickle=True)
    if mode == 'T5':

        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        data = np.load(data_path, allow_pickle=True).item()
        attack_data = {"label": data['label'], 'data': []}

        for i in tqdm.tqdm(range(len(data['data'])), desc='Preparing attack data'):
            if len(data['data'][i]) == 3:
                ref, btw, mod = data['data'][i]
            else:
                print('what')
                continue

            pos_tagged = nlp(mod)
            adv_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == 'VERB']
            adj_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == ('NOUN' or 'PROPN' or 'PRON')]
            for a in adv_idx:
                for adverb in adverbs:
                    tokens = insert_word(adverb, pos_tagged, a, tokenizer, ref, btw=btw)
                    attack_data['data'].append({
                        "input": tokens,
                        "original": ''.join(data['data'][i]),
                        "inserted": adverb,
                        "insert_type": 'ADV',
                        })

            for b in adj_idx:
                for adjective in adjectives:
                    tokens = insert_word(adjective, pos_tagged, b, tokenizer, ref, btw=btw)
                    attack_data['data'].append({
                        "input": tokens,
                        "original": ''.join(data['data'][i]),
                        "inserted": adjective,
                        "insert_type": 'ADJ',
                        })

        np.save(data_path.rsplit('/', 1)[0] + name + '.npy', attack_data, allow_pickle=True)


"""
prepare_attack("models/msrpc_bert_epoch=2-val_macro=0.8393.ckpt", "results/bert/msrpc/data.npy", 'bert')
prepare_attack("models/msrpc_T5_epoch=2-val_macro=0.8696.ckpt", "results/T5/msrpc/data.npy", 'T5')
prepare_attack("models/rte_bert_epoch=5-val_macro=0.6986.ckpt", "results/bert/rte/data.npy", 'bert')
prepare_attack("models/rte_T5_epoch=7-val_macro=0.7243.ckpt", "results/T5/rte/data.npy", 'T5')
prepare_attack("models/seb_bert_epoch=2-val_macro=0.7489.ckpt", "results/bert/seb/data.npy", 'bert')

prepare_attack("models/seb_T5_epoch=6-val_macro=0.7449.ckpt", "results/T5/seb/data.npy", 'T5')

prepare_attack("models/wic_bert_epoch=2-val_macro=0.8066.ckpt", "results/bert/wic/data.npy", 'bert')
prepare_attack("models/wic_T5_epoch=5-val_macro=0.7680.ckpt", "results/T5/wic/data.npy", 'T5')
"""
prepare_attack("models/mnli_bert_epoch=1-val_macro=0.8304.ckpt", "results/bert/mnli/original_data.npy", 'bert',
                                   'attack_data_dev_m')
prepare_attack("models/mnli_bert_epoch=1-val_macro=0.8304.ckpt", "results/bert/mnli/other_data.npy", 'bert',
                                   'attack_data_dev_mm')
prepare_attack("models/qqp_bert_epoch=4-val_macro=0.9037.ckpt", "results/bert/qqp/other_data.npy", 'bert', 'attack_data')


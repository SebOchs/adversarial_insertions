import torch
import numpy as np
import spacy
import tqdm
from transformers import T5Tokenizer, BertTokenizer
device = torch.device("cuda")

nlp = spacy.load('en_core_web_sm')




def prepare_attack(data_path, mode, name='attack_data'):



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


        np.save(data_path.rsplit('/', 1)[0] + '/' + name, attack_data, allow_pickle=True)
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

        np.save(data_path.rsplit('/', 1)[0] + '/' + name + '.npy', attack_data, allow_pickle=True)


prepare_attack("results/bert/mnli/matched/data.npy", 'bert')




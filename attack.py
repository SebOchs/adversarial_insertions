import torch
import numpy as np
from lit_Model import LitBERT, LitT5
import spacy
import time
import utils
import tqdm

device = torch.device("cuda")


def to_torch(x):
    return torch.tensor(x).unsqueeze(0).to(device)


def attack(path, attack_data, mode, goal=None, name='attack_results.npy'):
    data = np.load(attack_data, allow_pickle=True).item()
    if mode == 'bert':
        ckpt = LitBERT.load_from_checkpoint(path)
        model = ckpt.model
        model.cuda()
        model.eval()
        data_collector = {
            'confidence': {},
            'query': {},
            'success': {},
            'adversary_with_info': []
        }
        if goal is not None:
            goal = goal
        else:
            goal = 1

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(data['data'])), desc='Attacking'):
                data_instance = data['data'][i]
                batch = data_instance['input']
                result = model(input_ids=to_torch(batch.input_ids),
                               token_type_ids=to_torch(batch.token_type_ids),
                               attention_mask=to_torch(batch.attention_mask))
                data_collector['query'][data_instance['original']] = \
                    data_collector['query'].get(data_instance['original'], 0) + 1
                if torch.argmax(result.logits).to('cpu').item() == goal:
                    # collect all the data
                    data_collector['success'][data_instance['original']] = \
                        data_collector['success'].get(data_instance['original'], 0) + 1
                    data_collector['confidence'][data_instance['confidence']] = \
                        data_collector['confidence'].get(data_instance['confidence'], 0) + 1
                    data_collector['adversary_with_info'].append({'type': data_instance['insert_type'],
                                                                  'inserted': data_instance['inserted'],
                                                                  'original': data_instance['original'],
                                                                  'confidence': data_instance['confidence'],
                                                                  'adversary': ckpt.tokenizer.decode(batch.input_ids)
                                                                 .split(ckpt.tokenizer.pad_token)[0]})
        np.save(attack_data.rsplit('/', 1)[0] + '/' + name, data_collector, allow_pickle=True)

    if mode == 'T5':
        ckpt = LitT5.load_from_checkpoint(path)
        model = ckpt.model
        model.cuda()
        model.eval()
        data_collector = {
            'query': {},
            'success': {},
            'adversary_with_info': []
        }
        if goal is not None:
            goal = goal
        else:
            goal = 'True'

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(data['data'])), desc='Attacking'):
                data_instance = data['data'][i]
                batch = data_instance['input']
                result = ckpt.tokenizer.decode(model.generate(input_ids=to_torch(batch.input_ids),
                                                              attention_mask=to_torch(batch.attention_mask))[0],
                                               skip_special_tokens=True)
                data_collector['query'][data_instance['original']] = \
                    data_collector['query'].get(data_instance['original'], 0) + 1
                if result == goal:
                    # collect all the data
                    data_collector['success'][data_instance['original']] = \
                        data_collector['success'].get(data_instance['original'], 0) + 1
                    data_collector['adversary_with_info'].append({'type': data_instance['insert_type'],
                                                                  'inserted': data_instance['inserted'],
                                                                  'original': data_instance['original'],
                                                                  'adversary':
                                                                      ckpt.tokenizer.decode(batch.input_ids)
                                                                 .split(ckpt.tokenizer.pad_token)[0]})
        np.save(attack_data.rsplit('/', 1)[0] + '/' + name, data_collector, allow_pickle=True)


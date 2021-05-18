import torch
import numpy as np
from lit_Model import LitBERT, LitT5
import tqdm

device = torch.device("cuda")


def to_torch(x):
    """
    Prepares data for gpu
    :param x: A torch array on cpu or numpy array
    :return: torch tensor to gpu
    """
    return torch.tensor(x).unsqueeze(0).to(device)


def attack(path, attack_data, mode, goal=None, name='attack_results.npy'):
    """
    Executes attack on the BERT/T5 model with the prepared data and saves the results to a numpy file
    :param path: string / path of the model
    :param attack_data: string / path to prepared adversarial candidates
    :param mode: string / specify if model is T5 or bert
    :param goal: string or int / Name of the label adversarial examples should produce, based on model (int for bert,
    string for T5)
    :param name: string / name of the file the attack results should be saved to
    :return: Nothing
    """
    data = np.load(attack_data, allow_pickle=True).item()
    if mode == 'bert':
        # Load model
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
        # Set goal
        if goal is not None:
            goal = goal
        else:
            goal = 1

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(data['data'])), desc='Attacking'):
                # model prediction
                data_instance = data['data'][i]
                batch = data_instance['input']
                result = model(input_ids=to_torch(batch.input_ids),
                               token_type_ids=to_torch(batch.token_type_ids),
                               attention_mask=to_torch(batch.attention_mask))
                data_collector['query'][data_instance['original']] = \
                    data_collector['query'].get(data_instance['original'], 0) + 1
                # prediction should be goal
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
        # Load model
        ckpt = LitT5.load_from_checkpoint(path)
        model = ckpt.model
        model.cuda()
        model.eval()

        data_collector = {
            'query': {},
            'success': {},
            'adversary_with_info': []
        }
        # Set goal
        if goal is not None:
            goal = goal
        else:
            goal = 'True'

        with torch.no_grad():
            for i in tqdm.tqdm(range(len(data['data'])), desc='Attacking'):
                # model prediction
                data_instance = data['data'][i]
                batch = data_instance['input']
                result = ckpt.tokenizer.decode(model.generate(input_ids=to_torch(batch.input_ids),
                                                              attention_mask=to_torch(batch.attention_mask))[0],
                                               skip_special_tokens=True)
                data_collector['query'][data_instance['original']] = \
                    data_collector['query'].get(data_instance['original'], 0) + 1
                # prediction should be goal
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


attack("models/mnli_T5_epoch=0-val_macro=0.8107.ckpt", "results/T5/mnli/matched_attack_data.npy", 'T5',
       goal='entailment', name='matched_attack_results')
attack("models/mnli_T5_epoch=0-val_macro=0.8107.ckpt", "results/T5/mnli/mismatched_attack_data.npy", 'T5',
       goal='entailment', name='mismatched_attack_results')
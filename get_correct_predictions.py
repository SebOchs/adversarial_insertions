import torch
from torch.utils.data import Subset, DataLoader
from lit_Model import LitBERT, LitT5
from dataloading import MyBertDataset, MyT5Dataset
import numpy as np
import os


def save_correct_incorrect_predictions(path, mode, label=0, testdata="", to_save="custom_correct_predictions"):
    """
    Find and save the correct predictions of a model on a test set of the most negative label
    :param path: string / model path
    :param mode: string / bert or t5 model
    :param label: int or string / name of the label
    :param testdata: string / path to preprocessed test data folder if test data not in checkpoint
    :return:nothing
    """
    data_collector = {}
    device = torch.device("cuda")

    if mode == 'bert':
        # Load checkpoint and get necessary objects
        checkpoint = LitBERT.load_from_checkpoint(path)
        if len(testdata) > 0:
            test_set = MyBertDataset(testdata)
        else:
            test_set = checkpoint.test_data
        sub_set = Subset(test_set, [i for i in range(len(test_set)) if test_set[i][3] == label])
        model = checkpoint.model
        model.cuda()
        model.eval()
        tokenizer = checkpoint.tokenizer
        data_collector['label'] = label
        data_collector['length'] = len(sub_set)
        dataload = DataLoader(sub_set, batch_size=checkpoint.hparams['batch_size'], shuffle=False)

        correct_guesses = []
        confidence = []
        with torch.no_grad():
            for i in dataload:
                text, seg, att, lab = tuple(t.to(device) for t in i)
                results = model(input_ids=text, token_type_ids=seg, attention_mask=att, labels=lab)
                correct_guesses_idx = list((torch.argmax(results.logits, 1) == 0).nonzero().to('cpu').numpy().flatten())
                confidence.append(torch.nn.functional.softmax(results.logits, 1)[:, label][correct_guesses_idx])
                correct_guesses.append([tokenizer.decode(x).replace(tokenizer.cls_token, '')
                                       .split(tokenizer.sep_token)[:2] for x in i[0][correct_guesses_idx]])

            correct_guesses = [correct_guesses[x][y] for x in range(len(correct_guesses)) for y in
                               range(len(correct_guesses[x]))]
            confidence = list(torch.cat((confidence)).to('cpu').numpy())
            data_collector['data'] = list(zip(correct_guesses, confidence))
            data_collector['accuracy'] = len(confidence) / len(sub_set)
            where_to_save = 'results/' + path.split('/')[1].split('_')[1] + '/' + path.split('/')[1].split('_')[0]
            os.makedirs(where_to_save, exist_ok=True)
            if len(testdata) == 0:
                np.save(where_to_save + '/correct_predictions.npy', data_collector, allow_pickle=True)
            else:
                np.save(where_to_save + '/' + to_save, data_collector, allow_pickle=True)
    if mode == 'T5':
        # wic has different preprocessing than the rest, needs different string splitting
        if path.find('wic') != -1:
            splitter = 3
        else:
            splitter = 2
        # Load checkpoint and necessary data
        checkpoint = LitT5.load_from_checkpoint(path)
        if len(testdata) > 0:
            test_set = MyT5Dataset(testdata)
        else:
            test_set = checkpoint.test_data

        model = checkpoint.model
        model.cuda()
        model.eval()
        tokenizer = checkpoint.tokenizer
        sub_set = Subset(test_set, [i for i in range(len(test_set))
                                    if tokenizer.decode(test_set[i][2]).split(tokenizer.eos_token)[0] == label])
        data_collector['label'] = label
        data_collector['length'] = len(sub_set)
        dataload = DataLoader(sub_set, batch_size=checkpoint.hparams['batch_size'], shuffle=False)

        correct_guesses = []
        with torch.no_grad():
            for i in dataload:
                text, attn, lab, lab_attn = tuple(t.to(device) for t in i)
                results = [tokenizer.decode(x, skip_special_tokens=True)
                           for x in model.generate(input_ids=text, attention_mask=attn)]
                correct_guesses_idx = (np.array(results) == label).nonzero()[0].tolist()
                # remove data without second part when first part is too long
                split_input = [tokenizer.decode(x).split(tokenizer.eos_token)[:-1] for x in i[0][correct_guesses_idx]
                               if len(tokenizer.decode(x).split(tokenizer.eos_token)[:-1]) == splitter]
                if splitter == 3:
                    correct_guesses.append([[split_input[x][0], split_input[x][1] + split_input[x][2].split(':', 1)[0],
                                             split_input[x][2].split(':', 1)[1]] for x in range(len(split_input))])

                else:
                    correct_guesses.append([[split_input[x][0], split_input[x][1].split(':', 1)[0] + ':',
                                             split_input[x][1].split(':', 1)[1]] for x in range(len(split_input))])

            data_collector['data'] = [correct_guesses[x][y] for x in range(len(correct_guesses)) for y in
                                      range(len(correct_guesses[x]))]
            data_collector['accuracy'] = len(data_collector['data']) / len(sub_set)
            print(data_collector['accuracy'])
            where_to_save = 'results/' + path.split('/')[1].split('_')[1] + '/' + path.split('/')[1].split('_')[0]
            os.makedirs(where_to_save, exist_ok=True)
            # careful not to overwrite data TODO: more elegant data management
            if len(testdata) == 0:
                np.save(where_to_save + '/correct_predictions.npy', data_collector, allow_pickle=True)
            else:
                np.save(where_to_save + '/custom_correct_predictions.npy', data_collector, allow_pickle=True)


save_correct_incorrect_predictions("models/mnli_T5_epoch=0-val_macro=0.8107.ckpt", 'T5', label='neutral',
                                   testdata='datasets/preprocessed/T5/MNLI/dev_mm.npy')

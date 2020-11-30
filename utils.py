import numpy as np


def f1(pr, tr, class_num):
    """
    Calculates F1 score for a given class
    :param pr: list of predicted values
    :param tr: list of actual values
    :param class_num: indicates class
    :return: f1 score of class_num for predicted and true values in pr, tr
    """

    # Filter lists by class
    pred = [x == class_num for x in pr]
    truth = [x == class_num for x in tr]
    mix = list(zip(pred, truth))
    # Find true positives, false positives and false negatives
    tp = mix.count((True, True))
    fp = mix.count((False, True))
    fn = mix.count((True, False))
    # Return f1 score, if conditions are met
    if tp == 0 and fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    if tp == 0 and fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    if recall == 0 and precision == 0:
        return 0
    else:
        return 2 * recall * precision / (recall + precision)


def macro_f1(predictions, truth):
    """
    Calculates macro f1 score, where all classes have the same weight
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: macro f1 between model predictions and actual values
    """

    f1_0 = f1(predictions, truth, 0)
    f1_1 = f1(predictions, truth, 1)
    f1_2 = f1(predictions, truth, 2)
    if np.sum([x == 1 for x in truth]) == 0:
        return (f1_0 + f1_2) / 2
    else:
        return (f1_0 + f1_1 + f1_2) / 3


def weighted_f1(predictions, truth):
    """
    Calculates weighted f1 score, where all classes have different weights based on appearance
    :param predictions: logits of model predictions
    :param truth: list of actual values
    :return: weighted f1 between model predictions and actual values
    """

    weight_0 = np.sum([x == 0 for x in truth])
    weight_1 = np.sum([x == 1 for x in truth])
    weight_2 = np.sum([x == 2 for x in truth])
    f1_0 = f1(predictions, truth, 0)
    f1_1 = f1(predictions, truth, 1)
    f1_2 = f1(predictions, truth, 2)
    return (weight_0 * f1_0 + weight_1 * f1_1 + weight_2 * f1_2) / len(truth)


def decode(token_list, tokenizer, mode='torch'):
    """
    translates a list of tokens including two sentences/sequences into a readable string
    :param mode: flag for determining how to handle type of token list
    :param token_list: list of wordpiece tokens /
    :param tokenizer: huggingface tokenizer
    :return: string sequence 1, string sequence 2
    """
    if mode == 'torch':
        decoded = tokenizer.decode(token_list.squeeze().tolist())
    elif mode == 'list':
        decoded = tokenizer.decode(token_list.input_ids)
    # Clean-up
    x = decoded.replace(tokenizer.cls_token, '')
    ans_list = x.split(tokenizer.sep_token, 1)
    ans_list[1] = ans_list[1].replace(tokenizer.sep_token, '')
    ans_list[1] = ans_list[1].replace(tokenizer.pad_token, '')
    return ans_list[0].lstrip().rstrip(), ans_list[1].lstrip().rstrip()
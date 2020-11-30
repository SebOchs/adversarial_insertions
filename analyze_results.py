import numpy as np
import torch
from lit_BERT import LitBERT


adversaries = np.load('results/bert/sciEntsBank/adversaries.npy', allow_pickle=True).item()
query = np.load('results/bert/sciEntsBank/adversaries.npy', allow_pickle=True).item()
adversarial_successes = np.load('results/bert/sciEntsBank/adversarial_successes.npy', allow_pickle=True).item()
original_nr = len(adversaries.keys())

print('test')
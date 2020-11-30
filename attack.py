import torch
import numpy as np
from lit_BERT import LitBERT
import spacy
import time
import utils
device = torch.device("cuda")


def to_torch(x):
    return torch.tensor(x).unsqueeze(0)


def predict(x):
    """
    Inserts an adverb/adjective at specific place in a sentence and let's model predict
    :param word: string, adverb or adjective
    :param x: sentence/sequence preprocessed by spacy
    :param j: integer/index, where param word gets inserted
    :param model: finetuned huggingface model
    :param tokenizer: huggingface tokenizer
    :return: model prediction of modified sentence, modified sentence
    """
    return torch.argmax(model(input_ids=to_torch(x.input_ids[:128]).to(device),
                              token_type_ids=to_torch(x.token_type_ids[:128]).to(device),
                              attention_mask=to_torch(x.attention_mask[:128]).to(device))[0]).item()


# Load checkpoint and get necessary objects
checkpoint = LitBERT.load_from_checkpoint("models/seb_bert_epoch=4-val_macro=0.7680.ckpt")
model = checkpoint.model
model.cuda()
model.eval()
tokenizer = checkpoint.tokenizer
# only get incorrect data instances
attack_data = np.load('datasets/preprocessed/bert/sciEntsBank/attack_data.npy', allow_pickle=True)
# Data collectors
query = {}
successes = {}
adversaries = {}
for i in attack_data:
    adversaries[i[1]] = []
start = time.time()
ctr = 1
for date in attack_data:
    dict_entry = date[3] + ' : ' + date[2]
    query[dict_entry] = query.get(dict_entry, 0) + 1
    if predict(date[0]) == 2:
        ref_ans, stud_ans = utils.decode(date[0], tokenizer, mode='list')
        adversaries[date[1]].append(stud_ans)
        successes[dict_entry] = successes.get(dict_entry, 0) + 1
        print("Date nr. ", ctr, ", time passed: ", time.time() - start)
        ctr += 1
print("Done after ", time.time() - start, " seconds.")
np.save("results/bert/sciEntsBank/query.npy", query)
np.save("results/bert/sciEntsBank/adversarial_successes", successes)
np.save("results/bert/sciEntsBank/adversaries.npy", adversaries)

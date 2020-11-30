import torch
import numpy as np
from lit_BERT import LitBERT
import spacy
import time
import utils
device = torch.device("cuda")

nlp = spacy.load('en_core_web_sm')
words = np.load("top_adjectives_adverbs.npy", allow_pickle=True)
adverbs = words.item()['ADV']
adjectives = words.item()['ADJ']
attack_data = []


def to_torch(x):
    return torch.tensor(x).unsqueeze(0)


def insert_word(word, x, j, tokenizer, ref_answer):
    """
    Inserts an adverb/adjective at specific place in a sentence
    :param ref_answer: reference answer as string
    :param word: string, adverb or adjective
    :param x: sentence/sequence preprocessed by spacy
    :param j: integer/index, where param word gets inserted
    :param tokenizer: huggingface tokenizer
    :return: preprocessed data of adversarial sequence
    """
    adversarial_ans = x[:j].text + ' ' + word + ' ' + x[j:].text
    new_tokens = tokenizer(ref_answer, adversarial_ans, max_length=128, padding='max_length')
    return new_tokens





# Load checkpoint and get necessary objects
checkpoint = LitBERT.load_from_checkpoint("models/seb_bert_epoch=4-val_macro=0.7680.ckpt")
model = checkpoint.model
model.cuda()
model.eval()
tokenizer = checkpoint.tokenizer
# only get incorrect data instances
val_data = torch.load('datasets/preprocessed/bert/sciEntsBank/correct.pt')
val_data = [x for x in val_data if x[3] == 0]
attack_data = []
start = time.time()
for date in val_data:
    ref_ans, stud_ans = utils.decode(date[0], tokenizer)
    pos_tagged = nlp(stud_ans)
    adv_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == 'VERB']
    adj_idx = [x for x in range(len(pos_tagged)) if pos_tagged[x].pos_ == ('NOUN' or 'PROPN' or 'PRON')]
    for i in adv_idx:
        for adverb in adverbs:
            tokens = insert_word(adverb, pos_tagged, i, tokenizer, ref_ans)
            attack_data.append([tokens, stud_ans, adverb, 'ADV'])

    for j in adj_idx:
        for adjective in adjectives:
            tokens = insert_word(adverb, pos_tagged, i, tokenizer, ref_ans)
            attack_data.append([tokens, stud_ans, adjective, 'ADJ'])

print("Done after ", time.time() - start, " seconds.")
np.save("datasets/preprocessed/bert/sciEntsBank/attack_data.npy", attack_data, allow_pickle=True)

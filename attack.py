import torch
import numpy as np
from lit_BERT import LitBERT
import spacy
import time
nlp = spacy.load('en_core_web_sm')
words = np.load("top_adjectives_adverbs.npy", allow_pickle=True)
adverbs = words.item()['ADV']
adjectives = words.item()['ADJ']


def to_torch(date):
    return torch.tensor(date).unsqueeze(0)


def insert_and_predict(word, x, j, model, tokenizer):
    adversarial_ans = x[:j].text + ' ' + word + ' ' + x[j:].text
    new_tokens = tokenizer(ref_ans, adversarial_ans, max_length=128, padding='max_length')
    return torch.argmax(model(input_ids=to_torch(new_tokens.input_ids[:128]),
                              token_type_ids=to_torch(new_tokens.token_type_ids[:128]),
                              attention_mask=to_torch(new_tokens.attention_mask[:128]))[0]).item(), adversarial_ans


def decode(token_list, tokenizer):
    decoded = tokenizer.decode(token_list)
    # Clean-up
    x = decoded.replace(tokenizer.cls_token, '')
    ans_list = x.split(tokenizer.sep_token, 1)
    ans_list[1] = ans_list[1].replace(tokenizer.sep_token, '')
    ans_list[1] = ans_list[1].replace(tokenizer.pad_token, '')
    return ans_list[0].lstrip().rstrip(), ans_list[1].lstrip().rstrip()


checkpoint = LitBERT.load_from_checkpoint("models/bert_epoch=3-val_macro=0.7608.ckpt")
model = checkpoint.model
model.eval()
tokenizer = checkpoint.tokenizer
val_data = [x for x in checkpoint.val_data.dataset.data if x[3] == 0]
print("Nr. of incorrect instances: ", len(val_data))

query = {}
adv_success = {}
adj_success = {}
adversaries = {}
start = time.time()
ctr = 1
for i in val_data:
    label = i[3]
    pred = torch.argmax(model(input_ids=to_torch(i[0]), token_type_ids=to_torch(i[1]), attention_mask=to_torch(i[2]))
                        [0]).item()
    ref_ans, stud_ans = decode(i[0], tokenizer)
    adversaries[stud_ans] = []
    x = nlp(stud_ans)
    for j in range(len(x)):
        # attack verb
        if x[j].pos_ == 'VERB':
            for adverb in adverbs:
                new_pred, new_ans = insert_and_predict(adverb, x, j, model, tokenizer)
                query[adverb] = query.get(adverb, 0) + 1
                if new_pred == 2:
                    adversaries[stud_ans].append(new_ans)
                    adv_success[adverb] = adv_success.get(adverb, 0) + 1
                    print(stud_ans, " : ", new_ans)


        # attack noun
        if x[j].pos_ == ('NOUN' or 'PROPN' or 'PRON'):
            for adjective in adjectives:
                new_pred, new_ans = insert_and_predict(adjective, x, j, model, tokenizer)
                query[adjective] = query.get(adjective, 0) + 1
                if pred != new_pred:
                    adversaries[stud_ans].append(new_ans)
                    adj_success[adjective] = adj_success.get(adjective, 0) + 1
                    print(stud_ans, " : ", new_ans)

    print("Date nr. ", ctr, ", time passed: ", time.time() - start)
    ctr += 1
print("Done after ", time.time() - start, " seconds.")
np.save("query.npy", query)
np.save("adversarial_adverbs.npy", adv_success)
np.save("adversarial_adjectives.npy", adj_success)
np.save("adversaries.npy", adversaries)

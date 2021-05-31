import torch
from lit_Model import LitBERT
import numpy as np
import matplotlib.pyplot as plt


def plot(name, title, data):
    fig, ax = plt.subplots()
    ax.hist(data, bins=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                        0.85, 0.9, 0.95, 1.0], color='#005aa9')
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('# of Adversaries')

    ax.set_title(title)
    plt.savefig(name, dpi=300)
    ax.clear()
    fig.clear()


def confidence_plotting(ckpt, adv_examples):
    # load checkpoint
    checkpoint = LitBERT.load_from_checkpoint(ckpt)
    model = checkpoint.model
    tokenizer = checkpoint.tokenizer
    model.cuda()
    model.eval()

    # load adversarial examples
    adv_infos = np.load(adv_examples, allow_pickle=True).item()
    adversaries = [x['adversary'] for x in adv_infos['adversary_with_info']]

    confidence_scores_previous_label = []
    confidence_scores_new_label = []
    for adversary in adversaries:
        splitted = adversary.split(tokenizer.cls_token)[1].strip().split(tokenizer.sep_token)[:2]
        tokenized_input = tokenizer(splitted[0], splitted[1], return_tensors='pt')
        with torch.no_grad():
            output = model(input_ids=tokenized_input.input_ids.cuda(),
                           token_type_ids=tokenized_input.token_type_ids.cuda(),
                           attention_mask=tokenized_input.attention_mask.cuda())
            probabilities = torch.nn.functional.softmax(output[0].squeeze()).cpu().numpy()
            confidence_scores_previous_label.append(probabilities[0].item())
            confidence_scores_new_label.append(probabilities[-1].item())

    to_save = adv_examples.rsplit('/', 1)[0]
    plot(to_save + '/confidence_for_original.png',
         ' '.join(to_save.split("/")[1:]) + ': confidence scores for original label',
         confidence_scores_previous_label)
    plot(to_save + '/confidence_for_target',
         ' '.join(to_save.split("/")[1:]) + ': confidence scores for target label',
         confidence_scores_new_label)

"""
confidence_plotting('models/mnli_bert_epoch=1-val_macro=0.8304.ckpt',
                    'results/bert/mnli/matched/attack_results.npy')
confidence_plotting('models/mnli_bert_epoch=1-val_macro=0.8304.ckpt',
                    'results/bert/mnli/mismatched/attack_results.npy')
confidence_plotting('models/wic_bert_epoch=2-val_macro=0.8066.ckpt',
                    'results/bert/wic/attack_results.npy')
                    """
confidence_plotting('models/msrpc_bert_epoch=2-val_macro=0.8393.ckpt',
                    'results/bert/msrpc/attack_results.npy')
confidence_plotting('models/rte_bert_epoch=5-val_macro=0.6986.ckpt',
                    'results/bert/rte/attack_results.npy')
confidence_plotting('models/seb_bert_epoch=2-val_macro=0.7489.ckpt',
                    'results/bert/seb/ua/attack_results.npy')
confidence_plotting('models/seb_bert_epoch=2-val_macro=0.7489.ckpt',
                    'results/bert/seb/uq/attack_results.npy')
confidence_plotting('models/seb_bert_epoch=2-val_macro=0.7489.ckpt',
                    'results/bert/seb/ud/attack_results.npy')

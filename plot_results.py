import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import random


def plot(result_path, attack_data, mode):
    """
    Plots the confidence histogram of the adversarial examples and draws the reading examples
    :param result_path: string / path to the attack results
    :param attack_data: string / path to the correct incorrect prediction data
    :param mode: string / either bert or T5
    :return: nothing
    """

    def get_reading_ex(results):

        examples = []
        if mode == 'bert':
            for i in results:
                sent1, adversary = tuple([x.strip() for x in i['adversary'].split('[CLS]')[1].split('[SEP]')][:2])
                sent2 = i['original'].strip().split(sent1)[1].strip()
                examples.append({
                    'original_sent1': sent1,
                    'original_sent2': sent2,
                    'adversarial_sent2': adversary,
                    'inserted': i['inserted']
                })
        if mode == 'T5':
            for i in results:
                if i['original'].startswith('mnli:'):
                    sent1, sent2 = tuple(i['original'].split('mnli: premise:')[1].split('hypothesis:'))
                    adversary = i['adversary'].split('hypothesis:')[1].replace('</s>', '').strip()
                    examples.append({
                        'original_sent1': sent1,
                        'original_sent2': sent2,
                        'adversarial_sent2': adversary,
                        'inserted': i['inserted']
                    })
                elif i['original'].startswith('msrpc:'):
                    sent1, sent2 = tuple(i['original'].split('msrpc: sentence:')[1].split('paraphrase:'))
                    adversary = i['adversary'].split('paraphrase:')[1].replace('</s>', '').strip()
                    examples.append({
                        'original_sent1': sent1,
                        'original_sent2': sent2,
                        'adversarial_sent2': adversary,
                        'inserted': i['inserted']
                    })
                elif i['original'].startswith('rte:'):
                    sent1, sent2 = tuple(i['original'].split('rte: reference:')[1].split('answer:'))
                    adversary = i['adversary'].split('answer:')[1].replace('</s>', '').strip()
                    examples.append({
                        'original_sent1': sent1,
                        'original_sent2': sent2,
                        'adversarial_sent2': adversary,
                        'inserted': i['inserted']
                    })
                elif i['original'].startswith('asag:'):
                    sent1, sent2 = tuple(i['original'].split('asag: reference:')[1].split('student:'))
                    adversary = i['adversary'].split('student:')[1].replace('</s>', '').strip()
                    examples.append({
                        'original_sent1': sent1,
                        'original_sent2': sent2,
                        'adversarial_sent2': adversary,
                        'inserted': i['inserted']
                    })
                elif i['original'].startswith('wic:'):
                    sent1, sent2 = tuple(i['original'].split('first:')[1].split('second'))
                    adversary = i['adversary'].split('second')[1].replace('</s>', '').strip()
                    examples.append({
                        'original_sent1': sent1,
                        'original_sent2': sent2,
                        'adversarial_sent2': adversary,
                        'inserted': i['inserted']
                    })
                else:
                    raise ValueError('Not implemented for the current dataset.')

        return examples
    results = np.load(result_path, allow_pickle=True).item()
    data = np.load(attack_data, allow_pickle=True).item()
    reading = {}
    data_collector = {}
    if mode == 'bert':
        # histogram
        hist_data = []
        hist_2_data = data['confidences']
        for key in list(results['confidence'].keys()):
            for i in range(results['confidence'][key]):
                hist_data.append(key)
        # first plot
        fig, ax = plt.subplots()
        ax.hist(hist_data, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                color='#005aa9')
        ax.set_xlabel('Confidence score')
        ax.set_ylabel('# of Adversarial Examples')
        if len(result_path.split('/')) > 4:
            title = ' '.join(result_path.split('/')[1:4])
        else:
            title = ' '.join(result_path.split('/')[1:3])
        ax.set_title(title)
        plt.savefig(result_path.rsplit('/', 1)[0] + '/confidence_of_predictions_before_insertion.png', dpi=300)
        ax.clear()
        fig.clear()
        # second plot
        fig, ax = plt.subplots()
        ax.hist(hist_2_data, bins=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                color='#005aa9')
        ax.set_xlabel('Confidence score')
        ax.set_ylabel('# of Data Instances')
        if len(result_path.split('/')) > 4:
            title = ' '.join(result_path.split('/')[1:4])
        else:
            title = ' '.join(result_path.split('/')[1:3])
        ax.set_title(title)
        plt.savefig(result_path.rsplit('/', 1)[0] + '/confidence_of_predictions.png', dpi=300)
        ax.clear()
        fig.clear()
        # Analyze adversaries and get reading examples of top performing adv/adj
        data_1 = {}
        for key in list(results['success']):
            data_1[key] = results['success'][key] / results['query'][key]
        data_collector['success_rate_per_adversary'] = {k: v for k, v in sorted(data_1.items(),
                                                                                key=lambda item: item[1], reverse=True)}
        data_collector['new_accuracy'] = (len(data['data']) - len(data_1.keys())) / data['length']
        # Sort adverbs and adjectives by number of insertions
        adv = defaultdict(list)
        data_adv = Counter([x[0] for x in Counter([(x['inserted'], x['original'])
                                                   for x in results['adversary_with_info'] if x['type'] == 'ADV'])])
        for key, val in data_adv.items():
            adv[val].append(key)
        data_collector['adv_sorted'] = {k: v for k, v in sorted(adv.items(), key=lambda item: item[0], reverse=True)}
        adj = defaultdict(list)
        data_adj = Counter([x[0] for x in Counter([(x['inserted'], x['original'])
                                                   for x in results['adversary_with_info'] if x['type'] == 'ADJ'])])
        for key, val in data_adj.items():
            adj[val].append(key)
        data_collector['adj_sorted'] = {k: v for k, v in sorted(adj.items(), key=lambda item: item[0],
                                                                reverse=True)}
        # examples to check for adverbs
        for i in [item for sub_list in list(data_collector['adv_sorted'].values())[:10] for item in sub_list][:10]:
            # print('Adv: ', i)
            examples = [x for x in results['adversary_with_info'] if x['type'] == 'ADV'
                        and x['inserted'] == i]
            random.shuffle(examples)
            if len(examples) > 5:
                reading[i + '_adv'] = get_reading_ex(examples[:5])
            else:
                reading[i + '_adv'] = get_reading_ex(examples)

        # examples to check for adjectives
        for i in [item for sub_list in list(data_collector['adj_sorted'].values())[:10] for item in sub_list][:10]:
            # print('Adj: ', i)
            examples = [x for x in results['adversary_with_info'] if x['type'] == 'ADJ'
                        and x['inserted'] == i]
            random.shuffle(examples)
            if len(examples) > 5:
                reading[i + '_adj'] = get_reading_ex(examples[:5])
            else:
                reading[i + '_adj'] = get_reading_ex(examples)

        np.save(result_path.rsplit('/', 1)[0] + '/final_results.npy', data_collector, allow_pickle=True)
        np.save(result_path.rsplit('/', 1)[0] + '/reading.npy', reading, allow_pickle=True)
    if mode == 'T5':
        # print(result_path)
        # Analyze adversaries
        data_1 = {}
        for key in list(results['success']):
            data_1[key] = results['success'][key] / results['query'][key]
        data_collector['success_rate_per_adversary'] = {k: v for k, v in sorted(data_1.items(),
                                                                                key=lambda item: item[1], reverse=True)}
        data_collector['new_accuracy'] = (len(data['data']) - len(data_1.keys())) / data['length']
        adv = defaultdict(list)
        data_adv = Counter([x[0] for x in Counter([(x['inserted'], x['original'])
                                                   for x in results['adversary_with_info'] if x['type'] == 'ADV'])])
        for key, val in data_adv.items():
            adv[val].append(key)
        data_collector['adv_sorted'] = {k: v for k, v in sorted(adv.items(), key=lambda item: item[0], reverse=True)}
        adj = defaultdict(list)
        data_adj = Counter([x[0] for x in Counter([(x['inserted'], x['original'])
                                                   for x in results['adversary_with_info'] if x['type'] == 'ADJ'])])
        for key, val in data_adj.items():
            adj[val].append(key)
        data_collector['adj_sorted'] = {k: v for k, v in sorted(adj.items(), key=lambda item: item[0],
                                                                reverse=True)}
        # examples to check for adverbs
        for i in [item for sub_list in list(data_collector['adv_sorted'].values())[:10] for item in sub_list][:10]:
            # print('Adv: ', i)
            examples = [x for x in results['adversary_with_info'] if x['type'] == 'ADV'
                        and x['inserted'] == i]
            random.shuffle(examples)
            if len(examples) > 5:
                reading[i + '_adv'] = get_reading_ex(examples[:5])
            else:
                reading[i + '_adv'] = get_reading_ex(examples)

        # examples to check for adjectives
        for i in [item for sub_list in list(data_collector['adj_sorted'].values())[:10] for item in sub_list][:10]:
            # print('Adj: ', i)
            examples = [x for x in results['adversary_with_info'] if x['type'] == 'ADJ'
                        and x['inserted'] == i]
            random.shuffle(examples)
            if len(examples) > 5:
                reading[i + '_adj'] = get_reading_ex(examples[:5])
            else:
                reading[i + '_adj'] = get_reading_ex(examples)

        np.save(result_path.rsplit('/', 1)[0] + '/final_results.npy', data_collector, allow_pickle=True)
        np.save(result_path.rsplit('/', 1)[0] + '/reading.npy', reading, allow_pickle=True)


# bert

plot('results/bert/mnli/matched/attack_results.npy', 'results/bert/mnli/matched/correct_predictions.npy', 'bert')
plot('results/bert/mnli/mismatched/attack_results.npy', 'results/bert/mnli/mismatched/correct_predictions.npy', 'bert')
plot('results/bert/msrpc/attack_results.npy', 'results/bert/msrpc/custom_correct_predictions.npy', 'bert')
plot('results/bert/rte/attack_results.npy', 'results/bert/rte/custom_correct_predictions.npy', 'bert')
plot('results/bert/seb/ua/attack_results.npy', 'results/bert/seb/ua/correct_predictions.npy', 'bert')
plot('results/bert/seb/uq/attack_results.npy', 'results/bert/seb/uq/correct_predictions.npy', 'bert')
plot('results/bert/seb/ud/attack_results.npy', 'results/bert/seb/ud/correct_predictions.npy', 'bert')
plot('results/bert/wic/attack_results.npy', 'results/bert/wic/custom_correct_predictions.npy', 'bert')

# T5
plot('results/T5/mnli/matched_attack_results.npy', 'results/T5/mnli/correct_predictions.npy', 'T5')
plot('results/T5/mnli/mismatched_attack_results.npy', 'results/T5/mnli/custom_correct_predictions.npy', 'T5')
plot('results/T5/msrpc/attack_results.npy', 'results/T5/msrpc/data.npy', 'T5')
plot('results/T5/rte/attack_results.npy', 'results/T5/rte/data.npy', 'T5')
plot('results/T5/seb/ua/attack_results.npy', 'results/T5/seb/ua/data.npy', 'T5')
plot('results/T5/seb/uq/attack_results.npy', 'results/T5/seb/uq/data.npy', 'T5')
plot('results/T5/seb/ud/attack_results.npy', 'results/T5/seb/ud/data.npy', 'T5')
plot('results/T5/wic/attack_results.npy', 'results/T5/wic/data.npy', 'T5')



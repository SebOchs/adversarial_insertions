import nltk
import numpy as np
import os
import student_lab.criteria as criteria
from student_lab.lit_Model import LitBERT, LitT5
import tensorflow
import tensorflow_hub as hub
import time
import torch
import torch.nn.functional as F

tf = tensorflow.compat.v1
tf.disable_eager_execution()


def textfooler(data_loc, model_loc, adversary_goal):
    # Used for Text Fooler
    class USE(object):
        def __init__(self, cache_path):
            super(USE, self).__init__()
            os.environ['TFHUB_CACHE_DIR'] = cache_path
            module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
            self.embed = hub.Module(module_url)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.build_graph()
            self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

        def build_graph(self):
            self.sts_input1 = tf.placeholder(tf.string, shape=(None))
            self.sts_input2 = tf.placeholder(tf.string, shape=(None))

            sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
            sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
            self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
            clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
            self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

        def semantic_sim(self, sents1, sents2):
            scores = self.sess.run(
                [self.sim_scores],
                feed_dict={
                    self.sts_input1: sents1,
                    self.sts_input2: sents2,
                })
            return scores

    # Text Fooler function #1
    def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
        """
        embeddings is a matrix with (d, vocab_size)
        """
        sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values

    def predict(ckpt, ref, ans):
        def to_torch(ids):
            return torch.Tensor([ids]).long().cuda()

        tokenizer = ckpt.tokenizer
        encoded = tokenizer(ref, " ".join(ans))
        return ckpt.model(input_ids=to_torch(encoded.input_ids), attention_mask=to_torch(encoded.attention_mask),
                          token_type_ids=to_torch(encoded.token_type_ids)).logits.squeeze().cpu()

    # Text Fooler function #2
    def attack(text_ls, model, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
               import_score_threshold=-1., sim_score_threshold=0.7, sim_score_window=15, synonym_num=50,
               label_to_predict=1):
        adversaries = []
        if len(text_ls) != 2 or len(text_ls[1]) == 0:
            print("One input missing")
            return 0, 0, set()
        ref_ans, ans = text_ls
        to_modify = nltk.word_tokenize(ans)
        orig_logits = predict(model, ref_ans, to_modify)
        orig_probs = F.softmax(orig_logits, dim=0)
        orig_prob = orig_probs.max().item()
        orig_label = 0
        len_text = len(to_modify)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1

        # get the pos and verb tense info
        pos_ls = criteria.get_pos(to_modify)

        # get importance score
        leave_1_texts = [to_modify[:ii] + ['[UNK]'] + to_modify[min(ii + 1, len_text):] for ii in range(len_text)]
        leave_1_probs = []
        num_queries += len(leave_1_texts)

        for new_ans in leave_1_texts:
            new_logits = predict(model, ref_ans, new_ans)
            new_probs = F.softmax(new_logits, dim=0)
            leave_1_probs.append(new_probs)
        leave_1_probs = torch.stack(leave_1_probs)
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                  leave_1_probs_argmax))).data.cpu().numpy()

        # get words to perturb ranked by importance score for word in words_perturb
        words_perturb = []
        for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
            try:
                if score > import_score_threshold and to_modify[idx] not in stop_words_set:
                    words_perturb.append((idx, to_modify[idx]))
            except:
                print(idx, len(to_modify), import_scores.shape, to_modify, len(leave_1_texts))

        # find synonyms
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words, _ = pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)

        synonyms_all = []
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))

        # start replacing and attacking
        text_prime = to_modify[:]
        text_cache = text_prime[:]
        num_changed = 0
        for idx, synonyms in synonyms_all:
            new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
            new_probs = []
            new_labels = []
            for syn_text in new_texts:
                syn_logits = predict(model, ref_ans, syn_text)
                new_probs.append(F.softmax(syn_logits, dim=0))

            new_probs = torch.stack(new_probs)

            # compute semantic similarity
            if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                text_range_min = idx - half_sim_score_window
                text_range_max = idx + half_sim_score_window + 1
            elif idx < half_sim_score_window <= len_text - idx - 1:
                text_range_min = 0
                text_range_max = sim_score_window
            elif idx >= half_sim_score_window > len_text - idx - 1:
                text_range_min = len_text - sim_score_window
                text_range_max = len_text
            else:
                text_range_min = 0
                text_range_max = len_text
            semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                           list(map(lambda x: ' '.join(x[text_range_min:text_range_max]),
                                                    new_texts)))[
                    0]

            num_queries += len(new_texts)

            if len(new_probs.shape) < 2:
                new_probs = new_probs.unsqueeze(0)
            new_probs_mask = (label_to_predict == torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
            # prevent bad synonyms
            new_probs_mask *= (semantic_sims >= sim_score_threshold)
            # prevent incompatible pos (maybe not)

            synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]

            pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
            new_probs_mask *= pos_mask

            if np.sum(new_probs_mask) > 0:
                text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                num_changed += 1
                adversaries.append(tuple(text_prime))

        # Combine adversaries with new labels
        result = set(i for i in adversaries if i[0] != to_modify[:])
        return num_changed, num_queries, result

    # Data
    dataset = np.load(data_loc, allow_pickle=True).item()
    data = [x[0] for x in dataset['data']]
    data_length = dataset['length']
    old_accuracy = dataset['accuracy']
    ckpt = LitBERT.load_from_checkpoint(model_loc)
    ckpt.cuda()
    ckpt.eval()
    ckpt.freeze()

    # TextFooler building
    # prepare synonym extractor
    # build dictionary via the embedding file
    start = time.time()
    idx2word = {}
    word2idx = {}
    stop_words_set = criteria.get_stopwords()
    print("Building vocab...")
    with open("counter-fitted-vectors.txt", 'r', encoding="utf8") as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1
    print("Building cos sim matrix...")
    cos_sim = np.load("cos_sim_counter_fitting.npy", allow_pickle=True)
    print("Cos sim import finished!")
    use = USE("use")
    print('Start attacking!')
    result = {}
    adv_examples = []
    adversaries_ctr = 0
    queries = 0
    afflicted = 0
    ctr = 1
    for i in data:
        num_adversaries, num_queries, results = attack(i, ckpt, stop_words_set, word2idx, idx2word, cos_sim,
                                                       sim_predictor=use, label_to_predict=adversary_goal)
        adversaries_ctr += num_adversaries
        queries += num_queries

        if num_adversaries > 0:
            afflicted += 1
            adv_examples.append(["Reference: {}, Student: {}, Adversary: {}".format(i[0], i[1], " ".join(x))
                                 for x in results])
        print("Done with ", ctr)
        ctr += 1
    end = time.time()
    result['old_accuracy'] = old_accuracy
    result['loss'] = afflicted / data_length
    result['aaa'] = old_accuracy - result['loss']
    result['adv_ex'] = adversaries_ctr
    result['aff'] = afflicted
    result['time'] = (end - start) / 60
    result['results'] = adv_examples
    np.save(data_loc.rsplit('/', 1)[0] + '/textfooler_results', result)
    for text in ["{}: {}".format(x, result[x]) for x in result if x != 'results']:
        print(text)

textfooler("results/bert/msrpc/correct_predictions.npy",
           "models/msrpc_bert_epoch=2-val_macro=0.8393.ckpt", 1)

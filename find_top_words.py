import nltk
import numpy as np


def find_top_k_words_with_tag(k, tag):
    """
    Finds k top pos tagged words in the BROWN corpus based on bigrams
    :param k: int / number of top words
    :param tag: string / universal pos tag
    :return: list of strings
    """
    stop_words = nltk.corpus.stopwords.words('english')
    bigrams = nltk.bigrams((x[0].lower(), x[1]) for x in nltk.corpus.brown.tagged_words(tagset='universal'))
    # Filter bigrams
    tagged = []
    if tag == 'ADJ':
        next_tags = 'PROPN', 'NOUN', 'PRON'
        tagged = [x[0] for x in bigrams if x[0][1] == 'ADJ' and x[1][1] in next_tags]

    if tag == 'ADV':
        next_tags = 'VERB'
        tagged = [x[0] for x in bigrams if x[0][1] == 'ADV' and x[1][1] in next_tags]
        # tagged.extend([x[1] for x in bigrams if x[1][1] == 'ADV' and x[0][1] in next_tags])
    freq = nltk.FreqDist(x for x in tagged if x[0] not in stop_words) \
        .most_common(k)
    top_list = [x[0][0] for x in freq]

    return top_list


def best_words_percentile(words, percentage=0.7):
    """
    LEGACY from bachelor thesis project / finds best performing adjectives and adverbs during the attack
    based on percentile
    :param words: dict of strings as keys, ints as values / adjectives/adverbs are keys, values are number of adv.
    examples found on dev set inserting the specific adj/adv
    :param percentage: float / btw. 0 and 1, specifies percentile for top performing words
    :return: list of strings / best adv/adj of the attack in the given percentile
    """
    words = sorted(words.items(), key=lambda item: item[1], reverse=True)
    overall_sum = np.sum([x[1] for x in words])
    best = []
    iter_sum = 0
    for i in range(len(words)):
        if iter_sum <= overall_sum * percentage:
            best.append(words[i])
            iter_sum += words[i][1]
        else:
            break
    return [x[0] for x in best]


def main():
    top_adjectives = find_top_k_words_with_tag(100, 'ADJ')
    top_adverbs = find_top_k_words_with_tag(100, 'ADV')

    words = {'ADJ': top_adjectives, 'ADV': top_adverbs}
    np.save("top_adjectives_adverbs.npy", words)

    """
    not needed in this project, but maybe future work
    adj_res = np.load('adj_result.npy', allow_pickle=True).item()
    adv_res = np.load('adv_result.npy', allow_pickle=True).item()
    best_adjectives = best_words_percentile(adj_res)
    best_advberbs = best_words_percentile(adv_res)
    np.save('final_adj.npy', best_adjectives)
    np.save('final_adv.npy', best_advberbs)
    """


if __name__ == "__main__":
    main()
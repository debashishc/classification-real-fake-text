from metrics.Diversity import diversity
import nltk
from matplotlib import pyplot as plt

DATA_FILE = 'data/emnlp_news.txt'
TEST_FILE = 'data/test_emnlp.txt'

def get_sentences(filename):
    """ Return sentences given a text file.

        Need to tokenize
    """
    with open(filename, 'r') as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences

def find_plot_diversities(test_sentences, corpus_sentences, diversity_file):
    diversities = list()
    num_of_tests = len(test_sentences)
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in test_sentences[:num_of_tests]:
        diversities.append(diversity(sentence, corpus_sentences))

    # Minimum diversity can be used to then find the sentence and potentially
    # discover reasons causing diversity to decrease
    min_diversity = min(diversities)
    min_diversity_idx = diversities.index(min_diversity)
    print("Min diversity: {}".format(min_diversity))
    print("Sentence with min diversity: {}".format(test_sentences[min_diversity_idx]))
    # print("Novelties for {} sentences: \n {}".format(num_of_tests, diversities))
    
    with open(diversity_file, mode='w', encoding='utf-8') as f:
        f.write('all_diversities = \n')
        f.write('[')
        f.writelines(',\n'.join(str(nov) for nov in diversities))
        f.write(']')
    

    # plot diversities against sentence
    plt.plot(range(len(diversities)), diversities)
    plt.xlabel('Sentence')
    plt.ylabel('Novelty')
    plt.show()


if __name__ == '__main__':
    # save these sentences and diversities to save computation time
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences

    # save these sentences and novelties to save computation time
    corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences

    # find diversities within the corpus
    find_plot_diversities(corpus_sentences, corpus_sentences,     diversity_file='all_diversities_within_corpus.txt')
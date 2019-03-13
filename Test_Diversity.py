from metrics.Diversity import diversity
import nltk
from matplotlib import pyplot as plt

TEST_FILE = 'data/test_emnlp.txt'

def get_sentences(filename):
    """ Return sentences given a text file.

        Need to tokenize
    """
    with open(filename, 'r') as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences


if __name__ == '__main__':
    # save these sentences and diversities to save computation time
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences

    diversities = list()
    num_of_tests = len(test_sentences)
    for sentence in test_sentences[:num_of_tests]:
        diversities.append(diversity(sentence, test_sentences))

    # Minimum diversity can be used to then find the sentence and potentially
    # discover reasons causing diversity to decrease
    min_diversity = min(diversities)
    min_diversity_idx = diversities.index(min_diversity)
    print("Min diversity: {}".format(min_diversity))
    print("Sentence with min diversity: {}".format(
        test_sentences[min_diversity_idx]))
    # print("Diversities for {} sentences: \n {}".format(num_of_tests, diversities))

    diversity_file='all_diversities.txt'
    with open(diversity_file, mode='w', encoding='utf-8') as f:

        f.write('all_diversities = \n')
        f.write('[')
        f.writelines(',\n'.join(str(div) for div in diversities))
        f.write(']')
    

    plt.plot(range(len(diversities)), diversities)
    plt.xlabel('Sentence')
    plt.ylabel('Diversity')
    plt.show()



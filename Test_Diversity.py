from metrics.Diversity import diversity
import nltk
from matplotlib import pyplot as plt

TEST_FILE = 'data/test_emnlp.txt'

def get_sentences(filename):
    """ Return sentences given a text file.

        No need to re-tokenize 
    """
    with open(filename, 'r') as f:
        data = f.read()
    # sentences = nltk.sent_tokenize(data)
    return data


if __name__ == '__main__':
    # save these sentences and diversities to save computation time
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences

    diversities = list()
    num_of_tests = 100
    for sentence in test_sentences[:num_of_tests]:
        diversities.append(diversity(sentence, test_sentences))

    # Minimum diversity can be used to then find the sentence and potentially
    # discover reasons causing diversity to decrease
    print("Min diversity: {}".format(min(diversities)))
    print("Novelties for {} sentences: \n {}".format(num_of_tests, diversities))
    

    plt.plot(range(len(diversities)), diversities)
    plt.xlabel('Sentence')
    plt.ylabel('diversity')
    plt.show()



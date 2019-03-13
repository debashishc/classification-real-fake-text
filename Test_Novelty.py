from metrics.Novelty import novelty
import nltk
from matplotlib import pyplot as plt

DATA_FILE = 'data/emnlp_news.txt'
TEST_FILE = 'data/test_emnlp.txt'

def get_sentences(filename):
    """ Return sentences given a text file.
    """
    with open(filename, 'r') as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences


if __name__ == '__main__':
    # save these sentences and novelties to save computation time
    corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    print(len(corpus_sentences))
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    # print(test_sentences[9]) # They picked him off three times and kept him out of the end zone in a 22 - 6 victory at Arizona in 2013 .

    novelties = list()
    num_of_tests = 100
    for sentence in test_sentences[:num_of_tests]:
        novelties.append(novelty(sentence, corpus_sentences))

    # Minimum novelty can be used to then find the sentence and potentially
    # discover reasons causing novelty to decrease
    print("Min novelty: {}".format(min(novelties)))
    print("Novelties for {} sentences: \n {}".format(num_of_tests, novelties))
    

    # plot novelties against sentence
    plt.plot(range(len(novelties)), novelties)
    plt.xlabel('Sentence')
    plt.ylabel('Novelty')
    plt.show()



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

def find_plot_novelties(test_sentences, corpus_sentences, novelty_file):
    novelties = list()
    num_of_tests = len(test_sentences)
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in test_sentences[:num_of_tests]:
        novelties.append(novelty(sentence, corpus_sentences))

    # Minimum novelty can be used to then find the sentence and potentially
    # discover reasons causing novelty to decrease
    min_novelty = min(novelties)
    min_novelty_idx = novelties.index(min_novelty)
    print("Min novelty: {}".format(min_novelty))
    print("Sentence with min novelty: {}".format(test_sentences[min_novelty_idx]))
    # print("Novelties for {} sentences: \n {}".format(num_of_tests, novelties))
    
    with open(novelty_file, mode='w', encoding='utf-8') as f:
        f.write('all_novelties = \n')
        f.write('[')
        f.writelines(',\n'.join(str(nov) for nov in novelties))
        f.write(']')
    

    # plot novelties against sentence
    plt.plot(range(len(novelties)), novelties)
    plt.xlabel('Sentence')
    plt.ylabel('Novelty')
    plt.show()

if __name__ == '__main__':
    # save these sentences and novelties to save computation time
    corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    # print(len(corpus_sentences))
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    # print(test_sentences[9]) # They picked him off three times and kept him out of the end zone in a 22 - 6 victory at Arizona in 2013 .

    # find novelties within the corpus
    find_plot_novelties(corpus_sentences, corpus_sentences,     novelty_file='all_novelties_within_corpus.txt')



from metrics.Novelty import novelty
import nltk
from matplotlib import pyplot as plt
from tqdm import tqdm

def get_sentences(filename):
    """ Return sentences given a text file.
    """
    with open(filename, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences

def find_plot_novelties(test_sentences, corpus_sentences, novelty_file, metric):
    novelties = list()
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in tqdm(test_sentences, desc="Test sentences"):
        sent, _nov = novelty(sentence, corpus_sentences, metric)
        print('\n', sent)
        novelties.append(_nov)

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
    

    # # plot novelties against sentence
    # plt.plot(range(len(novelties)), novelties)
    # plt.xlabel('Sentence')
    # plt.ylabel('Novelty')
    # plt.show()

if __name__ == '__main__':

    DATA_FILE = 'data/emnlp_news.txt'
    TEST_FILE = 'data/test_emnlp.txt'
    GENERATED_FILE = 'data/generated_text3.txt'

    # save these sentences and novelties to save computation time
    # test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    generated_sentences = get_sentences(GENERATED_FILE) # 11055 sentences

    # save these sentences and novelties to save computation time
    real_sentences = get_sentences(DATA_FILE)  # 304222 sentences

    len_real = len(real_sentences)
    # print(len(real_sentences))

    find_plot_novelties(generated_sentences[5000:], real_sentences[:len_real//10],
                                novelty_file='extra/lev_novelties_real_5000_rest.txt', metric='levenshtein')

    # print(test_sentences[9]) # They picked him off three times and kept him out of the end zone in a 22 - 6 victory at Arizona in 2013 .

    # find novelties within the corpus
    # print("Novelties for first 3500")
    # print("Novelties for 3500 - 7000 sentences")
    # print("Novelties for 7000 - sentences")

    # find_plot_novelties(test_sentences[:100], real_sentences[:len_real//10], 
    #                     novelty_file='CHECK_lev_novelties_real_.txt')

    # find_plot_novelties(generated_sentences[:3500], corpus_sentences, 
    #                     novelty_file='LeakGAN_novelties_gen2_training_leven3500.txt', metric='levenshtein')
    # find_plot_novelties(generated_sentences[3500:7000], corpus_sentences, 
    #                     novelty_file='LeakGAN_novelties_gen2_training_leven7000.txt', metric='levenshtein')
    # find_plot_novelties(generated_sentences[7000:], corpus_sentences, 
                        # novelty_file='LeakGAN_novelties_gen2_training_leven10000.txt', metric='levenshtein')

    



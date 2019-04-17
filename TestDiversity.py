from metrics.Diversity import diversity
import nltk
from matplotlib import pyplot as plt
from tqdm import tqdm

DATA_FILE = 'data/emnlp_news.txt'
TEST_FILE = 'data/test_emnlp.txt'
GENERATED_FILE = 'data/generated_text2.txt'

def get_sentences(filename):
    """ Return sentences given a text file.
        The sentences will be tokenized in this function.
    """
    with open(filename, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences

def find_plot_diversities(test_sentences, corpus_sentences, diversity_file):
    diversities = list()
    num_of_tests = len(test_sentences)
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in tqdm(test_sentences[:num_of_tests], desc="Generated sentences"):
        # diversities.append(diversity(sentence, corpus_sentences, 'jaccard'))
        diversities.append(diversity(sentence, corpus_sentences, 'levenshtein'))


    # Minimum diversity can be used to then find the sentence and potentially
    # discover reasons causing diversity to decrease
    min_diversity = min(diversities)
    min_diversity_idx = diversities.index(min_diversity)
    print("Min diversity: {}".format(min_diversity))
    print("Sentence with min diversity: {}".format(test_sentences[min_diversity_idx]))
    # print("Novelties for {} sentences: \n {}".format(num_of_tests, diversities))
    
    with open(diversity_file, mode='w', encoding='utf-8') as f:
        f.write('all_diversities = \n')
        f.write('[\n')
        f.writelines(',\n'.join(str(div) for div in diversities))
        f.write('\n]')
    

    # # plot diversities against sentence
    # plt.plot(range(len(diversities)), diversities)
    # plt.xlabel('Sentence')
    # plt.ylabel('Novelty')
    # plt.show()


if __name__ == '__main__':
    # save these sentences and diversities to save computation time
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    generated_sentences = get_sentences(GENERATED_FILE) # 11055 sentences

    # save these sentences and novelties to save computation time
    real_sentences = get_sentences(DATA_FILE)  # 304222 sentences

    len_real = len(real_sentences)

    # find diversities within the corpus
    find_plot_diversities(test_sentences, real_sentences[:len_real//10],
                                diversity_file='lev_diversities_real.txt')



#     python Test_Diversity.py
# Example corpus sentence:  My sources have suggested that so far the company sees no reason to change its tax structures , which are perfectlyge its tax structures , which are perfectly legal .
# '
# Example test sentence:  a bathroom with a glass shower , sink and white .
# Min diversity: 0.7407407407407407
# Sentence with min diversity: a group of motorcycles parked on the sidewalks in a field .

# (tf-gpu) C:\Users\deb.chk\Documents\GitHub\NLP-tools>python Test_Diversity.py
# Example corpus sentence:  a bathroom with a glass shower , sink and white .
# Example test sentence:  a bathroom with a glass shower , sink and white .
# Generated sentences: 100%|███████████████████████████████████| 9003/9003 [04:33<00:00, 32.86it/s]
# Min diversity: 0.3125
# Sentence with min diversity: a bathroom with a mirrors reflection on far in the toilet .

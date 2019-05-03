# Import and download stopwords from NLTK.
import nltk
from nltk.corpus import stopwords
from nltk import download
download('stopwords')  # Download stopwords list.
import os
import gensim
import multiprocessing
from tqdm import tqdm
import numpy as np

from gensim.models import Word2Vec

# Using pre-trained word2vec Google News corpus (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
if not os.path.exists('../data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")

preloaded_model = gensim.models.KeyedVectors.load_word2vec_format(
    '../data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True)

# Normalizes the vectors in the word2vec class.
preloaded_model.init_sims(replace=True)


def get_sentences(filepath: str) -> list:
    """ Return sentences given a text file.
        The sentences will be tokenized in this function.
    """
    with open(filepath, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences


def preprocess(sentences: list) -> list:
    """ Take a list of sentences and return a list of list of words
    where all the words are alphabetic and not a stop word
    """
    stop_words = stopwords.words("english")

    preprocessed_sentences = list()
    for sentence in sentences:
        words = sentence.split()
        processed_words = [w for w in words if w.isalpha() and (not w in stop_words)]
        preprocessed_sentences.append(processed_words)
    return preprocessed_sentences


def find_plot_diversities(test_sentences, corpus_sentences, diversity_file):
    diversities = list()
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in tqdm(test_sentences, desc="Generated sentences"):
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
        f.write('[\n')
        f.writelines(',\n'.join(str(nov) for nov in diversities))
        f.write('\n]')


def diversity(sentence, tokenized_sentences) -> float:
    """ Calculate the diversity of sentence compared with a given corpus/document.
    """
    # sentences = nltk.sent_tokenize(document)

    min_edit_distance = np.inf
    for ref_sentence in tokenized_sentences:
        if sentence != ref_sentence:
            edit_distance = preloaded_model.wmdistance(sentence, ref_sentence)

            if edit_distance < min_edit_distance:
                min_edit_distance = edit_distance
                # maximum similarity is minimum edit distance

    return min_edit_distance


if __name__ == "__main__":

    # Datasets
    DATA_FILE = '../data/emnlp_news.txt'
    TEST_FILE = '../data/test_emnlp.txt'
    GENERATED_FILE = '../data/generated_text2.txt'

    processed_test_text = preprocess(get_sentences(TEST_FILE))

    # find_plot_diversities(processed_test_text, processed_test_text,
    #                             diversity_file='wmd_diversities_real_text.txt')

    find_plot_diversities(processed_test_text[:10], processed_test_text, diversity_file='../extra/wmd_diversities_real.txt')
    # python3 Test_Real_Diversity.py ; git add wmd_diversities_real_text.txt; git commit -m "Adding wmd_diversities_real.txt"; git push origin master


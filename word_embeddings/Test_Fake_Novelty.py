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
from matplotlib import pyplot as plt

from gensim.models import Word2Vec

# Using pre-trained word2vec Google News corpus (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
if not os.path.exists('../data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz'):
    raise ValueError("SKIP: You need to download the google news model")
    
preloaded_model = gensim.models.KeyedVectors.load_word2vec_format('../data/w2v_googlenews/GoogleNews-vectors-negative300.bin.gz', binary=True)

preloaded_model.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.


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


def find_plot_novelties(test_sentences, corpus_sentences, novelty_file):
    novelties = list()
    print("Example corpus sentence: ", corpus_sentences[0])
    print("Example test sentence: ", test_sentences[0])

    for sentence in tqdm(test_sentences, desc="Test sentences"):
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

    # # plot novelties against sentence
    # plt.plot(range(len(novelties)), novelties)
    # plt.xlabel('Sentence')
    # plt.ylabel('Novelty')
    # plt.show()


def novelty(sentence, tokenized_sentences) -> float:
    """ Calculate the diversity of sentence compared with a given corpus/document.
    """
    # sentences = nltk.sent_tokenize(document)

    min_edit_distance = np.inf
    for ref_sentence in tokenized_sentences:
        edit_distance = preloaded_model.wmdistance(sentence, ref_sentence)

        if edit_distance < min_edit_distance:
            min_edit_distance = edit_distance

    return min_edit_distance


if __name__ == "__main__":

    # Datasets

    DATA_FILE = '../data/emnlp_news.txt'
    GENERATED_FILE = '../data/generated_text3.txt'

    processed_fake_text = preprocess(get_sentences(GENERATED_FILE))
    processed_real_text = preprocess(get_sentences(DATA_FILE))

    len_real = len(processed_real_text)
    len_fake = len(processed_fake_text)

    # find_plot_novelties(processed_fake_text[:1500], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_1500.txt')
    # find_plot_novelties(processed_fake_text[1500:3000], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_15003000.txt')
    # find_plot_novelties(processed_fake_text[3000:4500], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_30004500.txt')
    # find_plot_novelties(processed_fake_text[4500:6000], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_45006000.txt')
    # find_plot_novelties(processed_fake_text[6000:7500], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_60007500.txt')
    # find_plot_novelties(processed_fake_text[7500:9000], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_75009000.txt')
    # find_plot_novelties(processed_fake_text[9000:10500], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_900010500.txt')
    # find_plot_novelties(processed_fake_text[10500:], processed_real_text[:len_real//10], novelty_file='../extra/wmd_novelties_fake3_10500rest.txt')
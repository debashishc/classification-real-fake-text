import os
import csv
import re
from nltk import sent_tokenize

from plotting import all_diversities as test_diversities
from plotting import all_novelties as test_novelties

def get_sentences(filepath):
    """ Return sentences given a text file.
    """
    # file_dir = os.path.dirname(os.path.realpath('__file__'))
    # filename = os.path.join(file_dir, filepath)
    # filename = os.path.abspath(os.path.realpath(filename))
    with open(filepath, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = sent_tokenize(data)
    return sentences

def label_sentence(sentence, label):
    return sentence, label

def process(sentences):
    new_sentences = list()
    for sentence in sentences:
        new_sentence = re.sub('[^a-zA-Z0-9\s]','',sentence)
        new_sentences.append(new_sentence)
    return new_sentences

def create_labelled_text(text, novs, divs, label):
    text_dict = dict()

    for ix, nov in enumerate(novs):
        text_dict[ix] = (text[ix], nov, divs[ix], label)
        
    return text_dict

def write_to_csv(text_dict, filename):
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "novelty", "diversity", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[1], values[2], values[3]]))
            for k, values in text_dict.items()]
        writer.writerows(data)


def read_list(filename: str) -> list:
    with open(file=filename, mode='r', encoding="ISO-8859-1") as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data[1:]:
            result_list.append(float(line.replace("]","")))
    return result_list

if __name__ == "__main__":
    # need to use a certain number of sentences each for classification
    DATA_FILE = "data/emnlp_news.txt"
    TEST_FILE = "data/test_emnlp.txt"
    GENERATED_FILE = "data/generated_text2.txt"

    print(__file__)

    # save these sentences and novelties to save computation time
    # corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    generated_sentences = get_sentences(GENERATED_FILE) # 11055 sentences

    # training the classifier
    # corpus_training = corpus_sentences[:6000]
    # print(corpus_training[:10]) # 10785
    test_training = test_sentences
    # print(corpus_training[:10])
    generated_training = generated_sentences


    # 
    gen_divs = read_list('Leakgan_diversities_intra_gen_jaccard.txt')
    gen_novs = read_list('LeakGAN_novelties_gen_training_jaccard.txt')

    print(len(generated_training), len(gen_divs), len(gen_novs))


    # create training csv
    test_dict = create_labelled_text(test_training, test_novelties, test_diversities, 1)
    write_to_csv(test_dict, filename='labelled_real_text.csv')

    gen_dict = create_labelled_text(generated_training, gen_novs, gen_divs, 0)
    write_to_csv(gen_dict, filename='labelled_fake_text.csv')
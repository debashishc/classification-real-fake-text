import os
import csv
import re
from nltk import sent_tokenize

def get_sentences(filepath):
    """ Return sentences given a text file.
    """
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(file_dir, filepath)
    filename = os.path.abspath(os.path.realpath(filename))
    with open(filename, 'r') as f:
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

def create_labelled_text(corpus, test, generated):
    text_dict = dict()

    for ix, sentence in enumerate(process(corpus)):
        text_dict[ix] = (sentence, 1)
    for ix, sentence in enumerate(process(test)):
        text_dict[ix] = (sentence, 1)
    for ix, sentence in enumerate(process(generated)):
        text_dict[ix] = (sentence, 0)
    return text_dict

def write_to_csv(text_dict, filename):
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "text", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[0], values[1]]))
            for k, values in text_dict.items()]
        writer.writerows(data)




if __name__ == "__main__":
    # need to use a certain number of sentences each for classification
    DATA_FILE = "data/emnlp_news.txt"
    TEST_FILE = "data/test_emnlp.txt"
    GENERATED_FILE = "data/generated_text.txt"

    # save these sentences and novelties to save computation time
    corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    generated_sentences = get_sentences(GENERATED_FILE) # 9003 sentences

    # training the classifier
    corpus_training = corpus_sentences[:4000]
    test_training = test_sentences[:4000]

    generated_training = generated_sentences[:8000]

    # testing the classifier
    test_test = test_training[4000:5003]
    generated_test = generated_sentences[8000:9003]

    # create training csv
    text_dict = create_labelled_text(corpus_training, test_training, generated_training)
    write_to_csv(text_dict, filename='labelled_text.csv')
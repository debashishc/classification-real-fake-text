import os
import csv
import re
from nltk import sent_tokenize

# from plotting import all_diversities as test_diversities
# from plotting import all_novelties as test_novelties

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

# no use of this yet
def label_sentence(sentence, label):
    return sentence, label

# no use of this yet
def process(sentences):
    new_sentences = list()
    for sentence in sentences:
        new_sentence = re.sub('[^a-zA-Z0-9\s]','',sentence)
        new_sentences.append(new_sentence)
    return new_sentences


def create_labelled_text(text, novs, divs, label):
    """ Create labelled text file with corresponding novelty and diversity values
    """
    text_dict = dict()

    for ix, t in enumerate(text):
        text_dict[ix] = (t, novs[ix], divs[ix], label)
        
    return text_dict


def write_to_csv(text_dict, filename):
    """
    """
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "text", "novelty", "diversity", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[0], values[1], values[2], values[3]]))
                for k, values in text_dict.items()]
        writer.writerows(data)

# bad function name
def create_labelled_files(text, label):
    """ Create labelled text file with no novelty or diversity values
    """
    text_dict = dict()

    for ix, t in enumerate(text):
        text_dict[ix] = (t, label)

    return text_dict


def write_to_file(text_dict, filename):
    """
    """
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "text", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[0], values[1]]))
                for k, values in text_dict.items()]
        writer.writerows(data)


def read_list(filename: str) -> list:
    """ Read list from a text file containing
    """
    import re
    with open(file=filename, mode='r', encoding="ISO-8859-1") as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data[0:]:
            result_list.extend(re.findall("\d+\.\d+", line))
    return result_list

if __name__ == "__main__":
    # need to use a certain number of sentences each for classification
    DATA_FILE = "data/emnlp_news.txt"
    TEST_FILE = "data/test_emnlp.txt"
    GENERATED_FILE = "data/generated_text3.txt"

    print(__file__)

    # save these sentences and novelties to save computation time
    # corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    real_sentences = get_sentences(TEST_FILE) # 10785 sentences
    fake_sentences = get_sentences(GENERATED_FILE)  # 11055 sentences

    # training the classifier
    # corpus_training = corpus_sentences[:6000]
    # print(corpus_training[:10]) # 10785
    # print(corpus_training[:10])


    #############################
    # Jaccard similarity
    ##############################

    fake_diversities = read_list('analysis_jaccard/jaccard_diversities_fake3.txt')
    fake_novelties = read_list('analysis_jaccard/jaccard_novelties_fake3.txt')

    real_novelties = read_list('analysis_jaccard/jaccard_novelties_real.txt')
    real_diversities = read_list(
        'analysis_jaccard/jaccard_diversities_real.txt')

    # label real text with 1
    # real_text_dict = create_labelled_text(
        # real_sentences, real_novelties, real_diversities, 1)
    # write_to_csv(real_text_dict, filename='labelled_real_text.csv')

    # label fake text with 0
    # fake_text_dict = create_labelled_text(
        # fake_sentences, fake_novelties, fake_diversities, 0)
    # write_to_csv(fake_text_dict, filename='labelled_fake_text.csv')
    print(len(fake_sentences), len(fake_diversities), len(fake_novelties))


    real_text = get_sentences(DATA_FILE)[:10000]
    fake_text = get_sentences('data/generated_text2.txt')

    real_dict = create_labelled_files(real_sentences, 1)
    write_to_file(real_dict, filename='unlabelled_real_text.csv')

    # label fake text with 0
    fake_dict = create_labelled_files(fake_sentences, 0)
    write_to_file(fake_dict, filename='unlabelled_fake_text.csv')
    

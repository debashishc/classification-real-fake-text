import os
import csv
import re
from nltk import sent_tokenize

def get_sentences(filepath):
    """ Return sentences given a text file.
    """
    with open(filepath, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = sent_tokenize(data)
    return sentences

def create_labelled_text(text, novs, divs, label):
    """ Create labelled text file with corresponding novelty and diversity values
    """
    text_dict = dict()

    for ix, t in enumerate(text):
        text_dict[ix] = (t, novs[ix], divs[ix], label)

    return text_dict


def create_labelled_dictionary(text, novelties, diversities, label):
    """ Create a dictionary with the keys being index with values of corresponding text,
    novelty, diversity and a label of 0 or 1.
    """
    text_dict = dict()

    for ix, t in enumerate(text):
        text_dict[ix] = (t, novelties[ix], diversities[ix], label)

    return text_dict


def write_to_csv(text_dict, filename):
    """ Given a dictionary containing text, novelties, diversities and label, this function
    can write to a csv with a given filename.
    """
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "text", "novelty", "diversity", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[0], values[1], values[2], values[3]]))
                for k, values in text_dict.items()]
        writer.writerows(data)


def read_list(filename: str) -> list:
    """ Read list from a text file containing
    """
    with open(file=filename, mode='r', encoding="ISO-8859-1") as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data:
            result_list.extend(re.findall("\d+\.\d+", line))
    return result_list

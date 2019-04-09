import csv, os
from nltk import sent_tokenize

def write_to_csv(text_dict: dict, filename: str) -> None:
    # real labelled as 1, fake labelled as 0
    fieldnames = ["index", "text", "label"]
    with open(file=filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        data = [dict(zip(fieldnames, [k, values[0], values[1]]))
                for k, values in text_dict.items()]
        writer.writerows(data)


def read_list_from_file(filename: str) -> list:
    with open(file=filename, mode='r') as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data[1:]:
            result_list.append(float(line.replace("]", "")))
    return result_list

def get_sentences(filepath: str) -> list:
    """ Return sentences given a text file.
    """
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(file_dir, filepath)
    filename = os.path.abspath(os.path.realpath(filename))
    with open(filename, 'r') as f:
        data = f.read()
    sentences = sent_tokenize(data)
    return sentences


if __name__ == "__main__":
    # need to use a certain number of sentences each for classification
    DATA_FILE = "data/emnlp_news.txt"
    TEST_FILE = "data/test_emnlp.txt"
    GENERATED_FILE = "data/generated_text2.txt"

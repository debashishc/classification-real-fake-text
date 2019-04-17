import gensim
print('gensim version: {}'.format(gensim.__version__))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import nltk
from tqdm import tqdm
# glove_model = gensim.models.KeyedVectors.load_word2vec_format('glove.6B.50d.txt')

DATA_FILE = 'data/emnlp_news.txt'
TEST_FILE = 'data/test_emnlp.txt'
GENERATED_FILE = 'data/generated_text2.txt'

def get_sentences(filename):
    """ Return sentences given a text file.
    """
    with open(filename, mode='r', encoding="ISO-8859-1") as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = "glove.6B.50d.txt"
tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

glove_model = model
gen = get_sentences(GENERATED_FILE)

all_tokens = []
diversities = list()
for sentence in gen:
    all_tokens.append(sentence.split())

for sentence in tqdm(gen):
    min_distance = np.inf
    tokens = sentence.split()
    for token, sent in zip(all_tokens, gen):
        if sentence != sent:
#             print('-'*50)
#             print(sentence)
#             print('Comparing to:', sent)
            edit_distance = glove_model.wmdistance(tokens, token)
#             print('distance = {}'.format(edit_distance))
            if edit_distance < min_distance:
                min_distance = edit_distance
                # maximum similarity is minimum edit distance
                max_sim = min_distance
                diversity = (1 - max_sim) / len(sent)
#     print('MAX_SIM = {}'.format(max_sim))
        


    diversities.append(diversity)


# Minimum diversity can be used to then find the sentence and potentially
# discover reasons causing diversity to decrease
min_diversity = min(diversities)
min_diversity_idx = diversities.index(min_diversity)
print("Min diversity: {}".format(min_diversity))
# print("Sentence with min diversity: {}".format(test_sentences[min_diversity_idx]))
# print("Novelties for {} sentences: \n {}".format(num_of_tests, diversities))

with open("test_glove.txt", mode='w', encoding='utf-8') as f:
    f.write('all_diversities = \n')
    f.write('[\n')
    f.writelines(',\n'.join(str(nov) for nov in diversities))
    f.write('\n]')
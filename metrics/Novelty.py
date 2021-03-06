
import collections
import nltk
from metrics.JaccardSimilarity import jaccard_similarity_words
from metrics.Levenshtein import levenshtein
import numpy as np

corpus = """
Monty Python (sometimes known as The Pythons) were a British surreal comedy group who created the sketch comedy show Monty Python's Flying Circus,
that first aired on the BBC on October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from the television series
into something larger in scope and impact, spawning touring stage shows, films, numerous albums, several books, and a stage musical.
The group's influence on comedy has been compared to The Beatles' influence on music."""

# we first tokenize the text corpus
tokens = nltk.word_tokenize(corpus)

def unigram(tokens):
    """Construct the unigram language model.
    """
    model = collections.defaultdict(lambda: 0.01)
    for token in tokens:
        try:
            model[token] += 1
        except KeyError:
            model[token] = 1
            continue
    for word in model:
        model[word] = model[word]/float(sum(model.values()))
    return model

# def novelty(sentence, tokenized_document):
#     """ Calculate the novelty of sentence compared with a given corpus/document.
#     """
#     # sentences = nltk.sent_tokenize(document)
#     sentences = tokenized_document
#     max_jaccard_sim = - float ('inf')

#     for ref_sentence in sentences:
#         jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
#         if jaccard_sim > max_jaccard_sim:
#             max_jaccard = jaccard_sim

#     return 1 - max_jaccard

from numba import cuda

def novelty(sentence: str, tokenized_sentences: str, similarity_metric: str) -> float:
    """ Calculate the novelty of sentence compared with a given corpus/document.
    """
    # sentences = nltk.sent_tokenize(document)
    max_sim_sentence = ''
    sentence = sentence.lower()
    tokenized_sentences = [sent.lower() for sent in tokenized_sentences]

    if similarity_metric == 'jaccard':
        max_sim = - np.inf
        
        for ref_sentence in tokenized_sentences:
            jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
            if jaccard_sim > max_sim:
                max_sim_sentence = ref_sentence
                max_sim = jaccard_sim

        return 1 - max_sim, max_sim_sentence

    elif similarity_metric == 'levenshtein':
        min_edit_distance = np.inf
        for ref_sentence in tokenized_sentences:
            edit_distance = levenshtein(sentence, ref_sentence) \
                                / max(len(sentence), len(ref_sentence))
            if edit_distance < min_edit_distance:
                max_sim_sentence = ref_sentence
                min_edit_distance = edit_distance

        return min_edit_distance, max_sim_sentence


if __name__ == '__main__':
    testset1 = "Monty"
    testset2 = "abracadabra gobbledygook rubbish"

    s1 = "The apple goes on trees."
    s2 = "The orange is not a fruit like apple"

    print(novelty (s2, corpus, 'jaccard'))


# https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity

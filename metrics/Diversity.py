import nltk
from metrics.JaccardSimilarity import jaccard_similarity_words
from metrics.Levenshtein import levenshtein
import numpy as np

# def diversity(sentence, tokenized_sentences):
#     """ Calculate the diversity of sentence compared with a given corpus/document.
#     """
#     # sentences = nltk.sent_tokenize(document)
#     max_jaccard_sim = - float('inf')

#     for ref_sentence in tokenized_sentences:
#         if sentence != ref_sentence:
#             jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
#             if jaccard_sim > max_jaccard_sim: <- HUGE BUG
#                 max_jaccard = jaccard_sim

#     return 1 - max_jaccard

def diversity(sentence: str, tokenized_sentences: str, similarity_metric: str) -> float:
    """ Calculate the diversity of sentence compared with a given corpus/document.
    """
    # sentences = nltk.sent_tokenize(document)

    if similarity_metric == 'jaccard':

        max_sim = -np.inf
        for ref_sentence in tokenized_sentences:
            if sentence != ref_sentence:
                jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
                if jaccard_sim > max_sim:
                    max_sim = jaccard_sim

        return 1 - max_sim

    elif similarity_metric == 'levenshtein':
        
        min_edit_distance = np.inf
        for ref_sentence in tokenized_sentences:
            if sentence != ref_sentence:
                edit_distance = levenshtein(sentence, ref_sentence) \
                                    / max(len(sentence), len(ref_sentence))

                if edit_distance < min_edit_distance:
                    min_edit_distance = edit_distance
                    # maximum similarity is minimum edit distance
                    # max_sim = min_edit_distance 

        return min_edit_distance

import nltk
from metrics.JaccardSimilarity import jaccard_similarity_words

def diversity(sentence, tokenized_sentences):
    """ Calculate the diversity of sentence compared with a given corpus/document.
    """
    # sentences = nltk.sent_tokenize(document)
    max_jaccard_sim = - float('inf')

    for ref_sentence in tokenized_sentences:
        if sentence != ref_sentence:
            jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
            if jaccard_sim > max_jaccard_sim:
                max_jaccard = jaccard_sim

    return 1 - max_jaccard


import collections
import nltk

from JaccardSimilarity import jaccard_similarity_words

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

def novelty(sentence, document):
    """ Calculate the novelty of sentence compared with a given corpus/document.
    """
    sentences = nltk.sent_tokenize(document)
    max_jaccard_sim = - float ('inf')

    for ref_sentence in sentences:
        jaccard_sim = jaccard_similarity_words(sentence, ref_sentence)
        if jaccard_sim > max_jaccard_sim:
            max_jaccard = jaccard_sim

    return 1 - max_jaccard


if __name__ == '__main__':
    testset1 = "Monty"
    testset2 = "abracadabra gobbledygook rubbish"

    s1 = "The apple goes on trees."
    s2 = "The orange is not a fruit like apple"

    print(novelty (s2, corpus))


# https://stackoverflow.com/questions/33266956/nltk-package-to-estimate-the-unigram-perplexity


def jaccard_similarity_words(sentence, ref_sentence):
    """Calculate the Jaccard Similarity between two sentences using sets of words.
    """
    set1 = set(sentence.split())  # split to get words
    set2 = set(ref_sentence.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    # print('set1: ', set1)
    # print('set2: ', set2)

    if len(union) == 0:
        return 1
    else:
        return len(intersection) / len(union)


def jaccard_similarity_chars(sentence, ref_sentence):
    """Calculate the Jaccard similarity between two sentences using sets of characters.
    """
    set1 = set(sentence)  # set casting on a string returns set of chars
    set2 = set(ref_sentence)
    intersection = set1.intersection(set2)
    union = set1.union(set2)


    if len(union) == 0:
        return 1
    else:
        return len(intersection) / len(union)

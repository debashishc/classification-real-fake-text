from metrics.Novelty import novelty
import nltk
from matplotlib import pyplot as plt

DATA_FILE = 'data/emnlp_news.txt'
TEST_FILE = 'data/test_emnlp.txt'

def get_sentences(filename):
    """ Return sentences given a text file.
    """
    with open(filename, 'r') as f:
        data = f.read()
    sentences = nltk.sent_tokenize(data)
    return sentences


if __name__ == '__main__':
    # # save these sentences and novelties to save computation time
    # corpus_sentences = get_sentences(DATA_FILE)  # 304222 sentences
    # # print(len(corpus_sentences))
    # test_sentences = get_sentences(TEST_FILE) # 10785 sentences
    # # print(test_sentences[9]) # They picked him off three times and kept him out of the end zone in a 22 - 6 victory at Arizona in 2013 .

    # novelties = list()
    # num_of_tests = 100 # len(test_sentences)
    # print("Example corpus sentence: ", corpus_sentences[0])
    # print("Example test sentence: ", test_sentences[0])
    # for sentence in test_sentences[:num_of_tests]:
    #     novelties.append(novelty(sentence, corpus_sentences))

    # # Minimum novelty can be used to then find the sentence and potentially
    # # discover reasons causing novelty to decrease
    # min_novelty = min(novelties)
    # min_novelty_idx = novelties.index(min_novelty)
    # print("Min novelty: {}".format(min_novelty))
    # print("Sentence with min novelty: {}".format(test_sentences[min_novelty_idx]))
    # print("Novelties for {} sentences: \n {}".format(num_of_tests, novelties))
    
    novelties = [0.9761904761904762, 0.9111111111111111, 0.9534883720930233, 0.926829268292683, 0.9512195121951219, 0.9375, 0.9, 0.9512195121951219, 0.8780487804878049, 0.85, 0.925, 0.9166666666666666, 0.9555555555555556, 0.9523809523809523, 0.9302325581395349, 0.9302325581395349, 0.9512195121951219, 0.9772727272727273, 0.925, 0.9534883720930233, 0.9347826086956522, 0.9736842105263158, 0.962962962962963, 0.8974358974358975, 0.9166666666666666, 0.9361702127659575, 0.8837209302325582, 0.9047619047619048, 0.9302325581395349, 0.9333333333333333, 0.9583333333333334, 0.9, 0.9, 0.9259259259259259, 0.9583333333333334, 0.9428571428571428, 0.967741935483871, 0.8780487804878049, 0.9318181818181819, 0.9523809523809523, 0.925, 0.9583333333333334, 0.9583333333333334, 0.868421052631579, 0.9259259259259259, 0.9523809523809523, 0.9574468085106383, 0.8378378378378378, 0.875, 0.9285714285714286, 0.9661016949152542, 0.9090909090909091,
                 0.9310344827586207, 0.9166666666666666, 0.9642857142857143, 0.9318181818181819, 0.9523809523809523, 0.9024390243902439, 0.9555555555555556, 0.9512195121951219, 0.8823529411764706, 0.9230769230769231, 0.9574468085106383, 0.9259259259259259, 0.868421052631579, 0.926829268292683, 0.925, 0.9361702127659575, 0.9090909090909091, 0.9782608695652174, 0.9347826086956522, 0.9555555555555556, 0.9622641509433962, 0.9607843137254902, 0.9591836734693877, 0.9782608695652174, 0.9285714285714286, 0.9074074074074074, 0.9069767441860466, 0.9117647058823529, 0.9523809523809523, 0.9285714285714286, 0.9545454545454546, 0.967741935483871, 0.9583333333333334, 0.9032258064516129, 0.8888888888888888, 0.9302325581395349, 0.8913043478260869, 0.9545454545454546, 0.9565217391304348, 0.9302325581395349, 0.9583333333333334, 0.8958333333333334, 0.9069767441860466, 0.9622641509433962, 0.9512195121951219, 0.9302325581395349, 0.8888888888888888, 0.8780487804878049]

    # plot novelties against sentence
    plt.plot(range(len(novelties)), novelties)
    plt.xlabel('Sentence')
    plt.ylabel('Novelty')
    plt.show()



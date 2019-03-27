#!/Users/dc/anaconda3/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from plotting import all_diversities as test_diversities
from plotting import all_novelties as test_novelties

def read_list(filename: str) -> list:
    with open(file=filename, mode='r') as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data[1:]:
            result_list.append(float(line.replace("]","")))
    return result_list


def normal_distribution(values: list, name_of_values: str) -> None:
    """
    normal_distribution(all_diversities, 'Diversity of generated text')
    """

    
    mean = np.nanmean(values)
    sigma = np.nanstd(values)

    num_bins = 100

    fig, ax = plt.subplots()

    #the histogram of the data
    n, bins, _ = ax.hist(values, num_bins, normed=True)

    # add a 'best fit' line
    y = mlab.normpdf(bins, mean, sigma)
    ax.plot(bins, y, linewidth=2, color='r')
    ax.set_xlabel(name_of_values)
    ax.set_xlim([0.5,1])
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of distribution: $\mu={}$, $\sigma={}$'.format(
        round(mean, 3), round(sigma, 3)))

    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #                          np.exp(- (bins - mean)**2 / (2 * sigma**2)),
    #                    linewidth=2, color='r')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()

def get_scatter_plot(data_x, data_y):
    """ Scatter plot of novelties vs diversities. """
    plt.scatter(test_diversities, test_novelties, color='red', label='Real text', alpha=0.3)
    plt.scatter(data_x, data_y, color='blue', label='Fake text', alpha=0.3)

    plt.title('{} vs {} of sentences for generated text set'.format("Novelties", "Diversities"))
    plt.xlabel('Diversity of sentence')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.ylabel('Novelty of sentence')
    plt.legend(loc='upper left')
    plt.show()
    # plt.savefig('scatter.png')

if __name__ == '__main__':
    fake_diversities = read_list('diversities_intra_gen.txt')
    fake_novelties = read_list('novelties_gen_training.txt')

    # test_diversities = read_list('diversities_intra_gen.txt')
    # test_novelties = read_list('diversities_intra_gen.txt')
    # all_diversities = read_list('all_diversities_within_corpus.txt')
    # print(len(all_diversities))
    # normal_distribution(all_diversities, 'Novelties of generated text')
    get_scatter_plot(fake_diversities, fake_novelties)


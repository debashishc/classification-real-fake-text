import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import re

def read_list(filename: str) -> list:
    """ Extract the diversity or novelty list from txt file
    
        Example usage:
        >>> read_list('jaccard_diversities_real.txt')
    """
    with open(file=filename, mode='r', encoding="ISO-8859-1") as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data:
            result_list.append(
                float(re.findall("\d+\.\d+", line)[0]) if line != 'inf' else 1.2760173970789273)

    return result_list


def normal_distribution(values: list, name_of_values: str) -> None:
    """
    Plot a normal distribution 
    
    Example usage:
    >>> normal_distribution(all_diversities, 'Diversity of generated text')
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
    ax.set_xlim([0,1])
    ax.set_ylabel('Probability density')
    ax.set_title(r'Histogram of distribution: $\mu={}$, $\sigma={}$'.format(
        round(mean, 3), round(sigma, 3)))

    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #                          np.exp(- (bins - mean)**2 / (2 * sigma**2)),
    #                    linewidth=2, color='r')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()


def get_scatter_plot(data_x, data_y, data_x1, data_y1):
    """
    Scatter plot of novelties vs diversities. 
    """
    plt.scatter(data_x, data_y, color='blue', label='Fake text', alpha=0.3)
    plt.scatter(data_x1, data_y1, color='red', label='Real text', alpha=0.3)

    plt.title('{} vs {} of real and fake sentences'.format(
        "Novelties", "Diversities"))
    plt.xlabel('Diversity of sentence')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
    plt.ylabel('Novelty of sentence')
    plt.legend(loc='upper left')
    plt.show()


def normalize(distances):
    """
    """
    min_val, max_val = min(distances), max(distances)
    return [(val - min_val) / (max_val - min_val) for val in distances]


def inverse_distances(distances):
    """
    Inverse distance to similaties [0, 0.5]
    """
    return [1 / (distance + 1) for distance in distances]

# def transform(distances):
#     d_max = max(distances)
#     return [ ( (d_max/ (0.001+distance)) - 1) / (d_max -1)  for distance in distances]


# def normed_dists_to_sims(distances):
#     """
#     Transform normalised distances to similarities [0, 1]
#     """
#     return [1 - distance for distance in distances]

def normed_dists_to_sims(distances):
    """
    Transform normalised distances to similarities [0, 1]
    """
    return distances

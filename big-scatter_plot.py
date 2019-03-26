#!/Users/dc/anaconda3/bin/python3

def read_list(filename: str) -> list:
    with open(file=filename, mode='r') as f:
        result_list = list()
        data = f.read().split(',\n')
        for line in data[1:]:
            result_list.append(float(line.replace("]","")))
    return result_list


all_diversities = read_list('all_diversities.txt')
# all_diversities = read_list('all_diversities_within_corpus.txt')
print(len(all_diversities))

def normal_distribution(values: list, name_of_values: str) -> None:
    """
    normal_distribution(all_diversities, 'Diversity of generated text')
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab
    
    mean = np.nanmean(values)
    sigma = np.nanstd(values)

    num_bins = 100

    fig, ax = plt.subplots()

    #the histogram of the data
    n, bins, patches = ax.hist(values, num_bins, normed=True)

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

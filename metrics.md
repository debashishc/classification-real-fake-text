# Metrics

## Perplexity

**TODO**: Some text here.

## Fluency

We use a language modeling toolkit - SRILM [Stolcke, 2002] to test the fluency
of generated sentences. SRILM calculates the perplexity of generated sentences
using the language model trained on respective corpus. We can see that C-GAN and
SVAE are not good at keeping the fluency of sentences. However, our model maintains
good fluency while generating texts of different sentiment labels, and it even
significantly outperforms the existing models on the small CR dataset.

## Novelty

Novelty can be used to investigate how different the generated sentences and the training corpus are. In other words, novelty can help figure out if the generator simply copies the sentence in the corpus instead of generating new ones. We calculate the novelty of each generated sentence $S_i$ as follows:

$$ Novelty (S_i)  = 1 - max \{\varphi (S_i, C_j) \}_{j = 1}^{j = \vert C \vert } $$

where $C$ is the sentence set of the training corpus and $\varphi$ is the Jaccard similarity function.

## Jaccard Similarity

Jaccard index is a metric often used for comparing similarity, dissimilarity, and distance of the data set. Measuring the Jaccard similarity coefficient between two data sets is the result of division between the number of features that are common to all divided by the number of properties as shown below.

$$ \varphi (S_1, S_2) = \frac{ \| S_1 \cup S_2 \| }{ \| S_1 \cap S_2 \|}  $$

## Diversity

We want to see if the generator can produce a variety of sentences. Given a collection of
generated sentences S, we define the diversity of sentences Si as follows:

$$ Diversity(S_i) = 1 - max \{ \varphi (S_i , S_j ) \}^{j = \vert S \vert , j \neq i}_{j=1} $$

where $\varphi$ is the Jaccard similarity function. We calculate the maximum Jaccard similarity between each sentence $S_i$ and other sentences in the collection.
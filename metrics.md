
# Metrics

**Perplexity**
    

**Fluency:**

    We use a language modeling toolkit - SRILM [Stolcke, 2002] to test the fluency
    of generated sentences. SRILM calculates the perplexity of generated sentences
    using the language model trained on respective corpus. We can see that C-GAN and
    SVAE are not good at keeping the fluency of sentences. However, our model maintains
    good fluency while generating texts of different sentiment labels, and it even
    significantly outperforms the existing models on the small CR dataset.

**Novelty:**

$Novelty (S_i)  = 1 - max \{\varphi (S_i, C_j) \}_{j = 1}^{j = \|C\|\}$
We want to investigate how different the generated sentences and the training corpus are.
In other words, we want to see if the generator simply copies the sentence in
the corpus instead of generating new ones. We calculate the novelty of each generated sentence
$S_i$ as follows:
where C is the sentence set of the training corpus, $$\varphi$$ is Jaccard similarity function. The average
values over generated sentences are shown in Table 2, we can see that RNNLM, SeqGAN and VAE are
not good at generating new sentences. On the contrary, our model performs exceptionally well, with the
ability to generate sentences different from that in the training corpus.

**Jaccard Similarity:**
    

**Diversity:**
    We want to see if the generator can produce a variety of sentences. Given a collection of
    generated sentences S, we define the diversity of sentences Si as follows:
        Diversity(Si) = 1 − max{ϕ(Si
        , Sj )}
        j=|S|,j6=i
        j=1 (11)
    where ϕ is the Jaccard similarity function. We calculate the maximum Jaccard similarity between
    each sentence $$S_i$$ and other sentences in the collection. The average values are shown in
    Table 3, and we can see that our model can generate a variety of sentences, while other models
    can not ensure the diversity of generated sentences.
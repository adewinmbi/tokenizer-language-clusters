# Tokenizer Language Clusters

Do tokenizers store cross-lingual grammatical concept representations?

```
H1: Same-concept cross-language clusters are tighter than
    different-concept cross-language clusters.
  same concept / cross-lang  avg cos sim: 0.5715
  diff concept / cross-lang  avg cos sim: 0.4745
  → SUPPORTED

H2: Same-concept cross-language clusters are tighter than
    different-concept within-language clusters.
  same concept / cross-lang  avg cos sim: 0.5715
  diff concept / same-lang   avg cos sim: 0.6191
  → NOT SUPPORTED
```

## Introduction

Brinkmann et al. provide evidence that ~7-8B LLMs have latent representations of grammatical concepts that are shared across typologically diverse languages ([2025](https://arxiv.org/pdf/2501.06346)). They do this through a multistep process: train a classifier to identify which grammatical concept is present, use attribution patching on SAE features to identify which features the classifier relies on most, identify overlap in these features across languages. In their results section, they say "This suggests that large language models—including those trained primarily on English—**learn** to rely on shared representations to detect particular concepts, rather than relying on language-specific representations." (Page 5). 

The implicit claim of the paper is that the model _learned_ cross-lingual abstractions through training. I want to answer the following question: can we identify whether this cross-lingual learning was actually done in the model, inherited from the tokenizer, or just a byproduct of the methodology?

The following experiment is motivated by the fact that the tokenizer is already shared across languages. It could already be clustering morphosyntactic features across languages (by having similar concepts be near each other in embedding space). Maybe all masculine words cluster in one location, and all feminine words cluster in another location regardless of language.

## Experiment

Part of my hypothesis is that the tokenizer is learning cross-lingual features by clustering the same concepts from typologically different languages near each other in embedding space. Let P and Q represent grammatical concepts, and A and B represent languages. To support the hypothesis, we should see that concept P in language A and concept P in language B are clustered closer than P in A and Q in B. It would also be interesting to see if P in A and P in B are closer than P in A and Q in A. This would be even stronger evidence showing that grammatical concepts cluster even more closely than words in the same language do in embedding space.

The Universal Dependencies treebank (used by Brinkmann et al.) already annotates each word with its morphosyntactic features. Since tokenization works on sub-words, we can "max pool", i.e., take the most common classifications from all the words the subword token appears in. Cluster inertia (sum of squared distances to the centroid) can be used to determine and compare how tight each cluster is.

## Results

See `results.json`.

## How to run

Download the Universal Dependencies and unzip `ud-treebanks-vx.x.tgz` into `\data\universal_dependencies`.

- Smoke test (English, French, German test split only): `python experiment.py --smoke`
- Full experiment (23 languages): `python experiment.py`
- With output saved to JSON: `python experiment.py --output results.json`

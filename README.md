# Tokenizer Language Clusters

Do tokenizers store cross-lingual grammatical concept representations?

Clusters of tokens in embedding space provide evidence towards: YES

```
========================================================================
HYPOTHESIS TESTS  (higher cosine sim = tighter cluster)
========================================================================

H1: Same-concept cross-language clusters are tighter than
    different-concept cross-language clusters.
  same concept / cross-lang  avg cos sim: 0.6307
  diff concept / cross-lang  avg cos sim: 0.5011
  → SUPPORTED

H2: Same-concept cross-language clusters are tighter than
    different-concept within-language clusters.
  same concept / cross-lang  avg cos sim: 0.6307
  diff concept / same-lang   avg cos sim: 0.5936
```

## Introduction

Brinkmann et al. provide evidence that ~7-8B LLMs have latent representations of grammatical concepts that are shared across typologically diverse languages ([2025](https://arxiv.org/pdf/2501.06346)). They do this through a multistep process: train a classifier to identify which grammatical concept is present, use attribution patching on SAE features to identify which features the classifier relies on most, identify overlap in these features across languages. In their results section, they say "This suggests that large language models—including those trained primarily on English—**learn** to rely on shared representations to detect particular concepts, rather than relying on language-specific representations." (Page 5). 

The implicit claim of the paper is that the model _learned_ cross-lingual abstractions through training. I want to answer the following question: can we identify whether this cross-lingual learning was actually done in the model, inherited from the tokenizer, or just a byproduct of the methodology?

The following experiment is motivated by the fact that the tokenizer is already shared across languages. It could already be clustering morphosyntactic features across languages (by having similar concepts be near each other in embedding space). Maybe all masculine words cluster in one location, and all feminine words cluster in another location regardless of language.

## Experiment

Part of my hypothesis is that the tokenizer is learning cross-lingual features by clustering the same concepts from typologically different languages near each other in embedding space. Let P and Q represent grammatical concepts, and A and B represent languages. To support the hypothesis, we should see that concept P in language A and concept P in language B are clustered closer than P in A and Q in B. It would also be interesting to see if P in A and P in B are closer than P in A and Q in A. This would be even stronger evidence showing that grammatical concepts cluster even more closely than words in the same language do in embedding space.

The Universal Dependencies treebank (used by Brinkmann et al.) already annotates each word with its morphosyntactic features. Since tokenization works on sub-words, we can "max pool", i.e., take the most common classifications from all the words the subword token appears in. Cluster inertia (sum of squared distances to the centroid) can be used to determine and compare how tight each cluster is.

## Results

```
========================================================================
HYPOTHESIS TESTS  (higher cosine sim = tighter cluster)
========================================================================

H1: Same-concept cross-language clusters are tighter than
    different-concept cross-language clusters.
  same concept / cross-lang  avg cos sim: 0.6307
  diff concept / cross-lang  avg cos sim: 0.5011
  → SUPPORTED

H2: Same-concept cross-language clusters are tighter than
    different-concept within-language clusters.
  same concept / cross-lang  avg cos sim: 0.6307
  diff concept / same-lang   avg cos sim: 0.5936

========================================================================
RESULTS BY FEATURE CATEGORY
========================================================================

  [Case]
    Same concept / cross-language
      cosine sim:       mean=0.4170  median=0.4170  n=1
      combined inertia: mean=246.7106  median=246.7106  n=1
    Diff concept / cross-language
      cosine sim:       mean=0.3551  median=0.3871  n=3
      combined inertia: mean=173.5530  median=237.2885  n=3
    Diff concept / same-language 
      cosine sim:       mean=0.8259  median=0.8247  n=6
      combined inertia: mean=365.8216  median=365.8081  n=6

  [Definite]
    Same concept / cross-language
      cosine sim:       mean=0.3852  median=0.3852  n=2
      combined inertia: mean=34.7131  median=34.7131  n=2
    Diff concept / cross-language
      cosine sim:       mean=0.4847  median=0.4847  n=2
      combined inertia: mean=33.7170  median=33.7170  n=2
    Diff concept / same-language 
      cosine sim:       mean=0.4006  median=0.4006  n=2
      combined inertia: mean=34.6895  median=34.6895  n=2

  [Degree]
    Diff concept / same-language 
      cosine sim:       mean=0.3615  median=0.3903  n=3
      combined inertia: mean=175.2711  median=257.8500  n=3

  [Gender]
    Same concept / cross-language
      cosine sim:       mean=0.8926  median=0.8926  n=2
      combined inertia: mean=855.3899  median=855.3899  n=2
    Diff concept / cross-language
      cosine sim:       mean=0.8793  median=0.8762  n=4
      combined inertia: mean=812.7952  median=827.5239  n=4
    Diff concept / same-language 
      cosine sim:       mean=0.9273  median=0.9253  n=4
      combined inertia: mean=679.3997  median=527.3098  n=4

  [Mood]
    Same concept / cross-language
      cosine sim:       mean=0.6601  median=0.8129  n=4
      combined inertia: mean=233.7094  median=296.0943  n=4
    Diff concept / cross-language
      cosine sim:       mean=0.3885  median=0.4199  n=8
      combined inertia: mean=132.0208  median=152.0801  n=8
    Diff concept / same-language 
      cosine sim:       mean=0.4903  median=0.5386  n=3
      combined inertia: mean=170.0210  median=157.8360  n=3

  [NumType]
    Same concept / cross-language
      cosine sim:       mean=0.8663  median=0.8663  n=1
      combined inertia: mean=55.1940  median=55.1940  n=1

  [Number]
    Same concept / cross-language
      cosine sim:       mean=0.8752  median=0.9002  n=6
      combined inertia: mean=1013.1642  median=975.8507  n=6
    Diff concept / cross-language
      cosine sim:       mean=0.8593  median=0.8692  n=6
      combined inertia: mean=1053.9440  median=1085.7967  n=6
    Diff concept / same-language 
      cosine sim:       mean=0.9063  median=0.9108  n=3
      combined inertia: mean=1087.6215  median=1014.6755  n=3

  [Number[psor]]
    Same concept / cross-language
      cosine sim:       mean=0.2901  median=0.2901  n=2
      combined inertia: mean=11.3632  median=11.3632  n=2
    Diff concept / cross-language
      cosine sim:       mean=0.2722  median=0.2722  n=2
      combined inertia: mean=11.3861  median=11.3861  n=2
    Diff concept / same-language 
      cosine sim:       mean=0.2965  median=0.2965  n=2
      combined inertia: mean=11.3405  median=11.3405  n=2

  [Person]
    Same concept / cross-language
      cosine sim:       mean=0.6502  median=0.6784  n=7
      combined inertia: mean=162.5660  median=59.6745  n=7
    Diff concept / cross-language
      cosine sim:       mean=0.5454  median=0.5801  n=14
      combined inertia: mean=142.3780  median=159.6739  n=14
    Diff concept / same-language 
      cosine sim:       mean=0.6104  median=0.6487  n=7
      combined inertia: mean=148.4722  median=161.2759  n=7

  [Person[psor]]
    Diff concept / same-language 
      cosine sim:       mean=0.2688  median=0.2688  n=1
      combined inertia: mean=10.1148  median=10.1148  n=1

  [Polarity]
    Same concept / cross-language
      cosine sim:       mean=0.4879  median=0.4879  n=1
      combined inertia: mean=14.7123  median=14.7123  n=1

  [Poss]
    Same concept / cross-language
      cosine sim:       mean=0.2989  median=0.2989  n=1
      combined inertia: mean=21.7196  median=21.7196  n=1

  [PronType]
    Same concept / cross-language
      cosine sim:       mean=0.4267  median=0.4782  n=9
      combined inertia: mean=36.9139  median=34.5185  n=9
    Diff concept / cross-language
      cosine sim:       mean=0.3183  median=0.3095  n=54
      combined inertia: mean=30.7712  median=27.4616  n=54
    Diff concept / same-language 
      cosine sim:       mean=0.4229  median=0.4477  n=28
      combined inertia: mean=33.9308  median=33.4508  n=28

  [Reflex]
    Same concept / cross-language
      cosine sim:       mean=0.5230  median=0.5230  n=1
      combined inertia: mean=10.8653  median=10.8653  n=1

  [Tense]
    Same concept / cross-language
      cosine sim:       mean=0.7817  median=0.7774  n=6
      combined inertia: mean=234.0979  median=228.9204  n=6
    Diff concept / cross-language
      cosine sim:       mean=0.6562  median=0.6192  n=14
      combined inertia: mean=168.1954  median=154.3580  n=14
    Diff concept / same-language 
      cosine sim:       mean=0.7022  median=0.6782  n=8
      combined inertia: mean=184.7423  median=179.4796  n=8

  [Typo]
    Same concept / cross-language
      cosine sim:       mean=0.2505  median=0.2505  n=1
      combined inertia: mean=10.3558  median=10.3558  n=1

  [VerbForm]
    Same concept / cross-language
      cosine sim:       mean=0.7459  median=0.7584  n=9
      combined inertia: mean=227.2376  median=253.3309  n=9
    Diff concept / cross-language
      cosine sim:       mean=0.7197  median=0.7340  n=24
      combined inertia: mean=213.5188  median=219.0628  n=24
    Diff concept / same-language 
      cosine sim:       mean=0.7967  median=0.8022  n=12
      combined inertia: mean=212.9036  median=209.4853  n=12

  [Voice]
    Same concept / cross-language
      cosine sim:       mean=0.3909  median=0.3909  n=1
      combined inertia: mean=30.0765  median=30.0765  n=1

```

## How to run

- Smoke test (English, French, German test split only): `python experiment.py --smoke`
- Full experiment (23 languages): `python experiment.py`
- With output saved to JSON: `python experiment.py --output results.json`

---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Sequence ML models

`kmer.models` connects encoders, distances, kernels, and background models to
scikit-learn-style estimators. The model should match the representation and
the scale of the dataset.

<!-- Illustration placeholder; replace with a user-supplied asset later.
```{image} ../assets/guide/models.svg
:alt: Sequence examples branch into linear and kernel decision boundaries, a nearest-neighbor neighborhood, and ranked differential k-mer scores.
:width: 560px
:align: center
:class: guide-illustration
```
-->

## Learning goals

By the end of this guide, you will be able to:

- choose among multinomial Naive Bayes, linear regression, SVM, and nearest-neighbor models;
- fit each model directly from sequence collections;
- interpret decision scores and differential k-mer weights;
- avoid common train/test and probability-estimation mistakes.

## Choose a model

| Model | Input strategy | Good starting point when |
|---|---|---|
| {class}`~kmer.models.KmerMultinomialNB` | Positive/negative k-mer counts | Classification and interpretable k-mer enrichment are both required |
| {class}`~kmer.models.KmerLinearRegression` | Sparse encoder features | The response is continuous and additive k-mer effects are interpretable |
| {class}`~kmer.models.LinearSVM` | Sparse encoder features | The dataset is large and additive k-mer effects are sufficient for classification |
| {class}`~kmer.models.KernelSVM` | GKM or transformed distance matrix | Pairwise sequence similarity captures interactions better |
| {class}`~kmer.models.KNNClassifier` | Proper sequence distance | Local neighborhoods are meaningful and the training set is moderate |

## Example data

The examples use a deliberately small dataset to illustrate APIs rather than
to estimate generalization performance.

```{code-cell} ipython3
import numpy as np

positives = [
    "ACGTACGTACGT",
    "ACGTACGTACGA",
    "AACGTACGTACG",
    "ACGTACGTTCGT",
]
negatives = [
    "TTTTAAAACCCC",
    "TTTAAAACCCCG",
    "GGGGCCCCAAAA",
    "TATATATATATA",
]
sequences = positives + negatives
y = np.array([1] * len(positives) + [0] * len(negatives))
```

## Multinomial Naive Bayes on k-mer counts

{class}`~kmer.models.KmerMultinomialNB` fits a binary classifier and exposes
the fitted per-feature log odds between positive and negative collections.

```{code-cell} ipython3
from kmer.encoders import SpectrumEncoder
from kmer.models import KmerMultinomialNB

nb = KmerMultinomialNB(
    featurizer=SpectrumEncoder(k=3, canonical_rc=True),
    alpha=1.0,
)
nb.fit(positives, negatives)

nb_queries = ["ACGTACGTACGG", "TTTTAAAACCCG"]
top_kmers = sorted(
    nb.kmer_scores_.items(),
    key=lambda item: item[1],
    reverse=True,
)[:5]
print("top enriched k-mers:", [
    (name, round(score, 3)) for name, score in top_kmers
])
print("query scores:", np.round(nb.decision_function(nb_queries), 3))
```

If explicit negatives are unavailable, pass a
{class}`~kmer.background.BaseBackgroundModel`; one negative is generated per
positive during fitting.

```{code-cell} ipython3
from kmer.background import KmerShuffler

background_nb = KmerMultinomialNB(
    featurizer=SpectrumEncoder(k=3),
    background=KmerShuffler(k=1, seed=42, endpoints="preserve"),
)
background_nb.fit(positives)
print("generated negatives:", background_nb.n_negatives_)
```

The classifier's sigmoid probabilities are convenient transformations of
summed log odds; they are not a substitute for external calibration when
calibrated probabilities are scientifically important.

## Linear regression on sparse k-mer counts

{class}`~kmer.models.KmerLinearRegression` fits an encoder and delegates
continuous-value prediction to scikit-learn's `LinearRegression`. Here the
illustrative response is each sequence's GC fraction.

```{code-cell} ipython3
from kmer.encoders import SpectrumEncoder
from kmer.models import KmerLinearRegression

activity = np.array([
    (seq.count("G") + seq.count("C")) / len(seq)
    for seq in sequences
])
regression = KmerLinearRegression(
    encoder=SpectrumEncoder(k=1),
    standardize=True,
    with_mean=False,
    fit_intercept=True,
)
regression.fit(sequences, activity)
print("targets:", np.round(activity, 3))
print("predictions:", np.round(regression.predict(sequences), 3))
print("coefficient shape:", regression.coef_.shape)
```

With `standardize=True`, `with_mean=False` scales variances without centering,
so sparse encoder matrices remain sparse. Set `with_mean=True` when centering
is required; sparse encoder output is then converted to a dense array and may
consume substantially more memory. Fit-time `sample_weight` is supported.
Other keyword arguments are forwarded to the installed scikit-learn
`LinearRegression`, allowing options such as `fit_intercept`, `copy_X`,
`n_jobs`, and `positive` when supported by that version. Some scikit-learn
versions require dense input for `positive=True`, so that option may not accept
sparse encoder output.

## SVM models

### Linear SVM on sparse k-mer counts

{class}`~kmer.models.LinearSVM` owns its encoder and applies the same feature
mapping during fitting and prediction.

```{code-cell} ipython3
from kmer.encoders import SpectrumEncoder
from kmer.models import LinearSVM

linear = LinearSVM(
    encoder=SpectrumEncoder(k=3, canonical_rc=True),
    C=1.0,
)
linear.fit(sequences, y)
print("training predictions:", linear.predict(sequences))
print("decision scores:", np.round(linear.decision_function(sequences), 3))
```

Set `standardize=True` when feature scales differ substantially. Sparse inputs
are scaled without mean centering. The `coef_` property exposes one weight per
encoder feature, allowing important words to be mapped back with
`get_feature_names_out()`.

```{code-cell} ipython3
names = linear.encoder.get_feature_names_out()
weights = linear.coef_[0]
top = np.argsort(weights)[-5:][::-1]
print([(names[i], round(float(weights[i]), 3)) for i in top])
```

### Kernel SVM on sequence similarity

{class}`~kmer.models.KernelSVM` accepts either a GKM-style object or a
{class}`~kmer.distance.DistanceKernel`. It handles train/train and
query/train matrix construction internally.

```{code-cell} ipython3
from kmer.kernels import GKMKernel
from kmer.models import KernelSVM

kernel_svm = KernelSVM(
    GKMKernel(),
    C=1.0,
)
kernel_svm.fit(sequences, y)
print("training predictions:", kernel_svm.predict(sequences))
print("decision scores:", np.round(kernel_svm.decision_function(sequences), 3))
```

Construct with `probability=True` before fitting if `predict_proba` is needed.
Probability estimation is slower and should be assessed for calibration on
held-out data.

## Nearest neighbors from a distance

```{code-cell} ipython3
from kmer.distance import Levenshtein
from kmer.models import KNNClassifier

knn = KNNClassifier(
    distance=Levenshtein(),
    n_neighbors=3,
    weights="distance",
)
knn.fit(sequences, y)
queries = ["ACGTACGTACGG", "TTTTAAAACCCG"]
print("predictions:", knn.predict(queries))
print("probabilities:\n", np.round(knn.predict_proba(queries), 3))
```

KNN computes a query-by-training distance matrix at prediction time. Use a
proper distance such as Hamming or Levenshtein; raw alignment scores have the
opposite direction. For large reference collections, consider
{class}`kmer.ann.ANNIndex`.

## Evaluation guidance

- Split biological groups before fitting encoders, choosing parameters, or
  generating related background sequences.
- Report held-out metrics rather than the training outputs shown here; use regression metrics such as MAE or $R^2$ for continuous responses.
- Use stratification for imbalanced labels and group-aware splits for related
  sequences.
- Tune `C`, representation parameters, and background choices inside the
  training data only.
- Prefer `LinearSVM` as a scalable baseline before adopting a more expensive
  pairwise model.

## Next steps

- Review representations in {doc}`encoders`, {doc}`distances`, and
  {doc}`kernels`.
- Design negative controls with {doc}`background`.
- Scale neighbor retrieval with {doc}`ann`.
- Consult {doc}`../api/models` for estimator signatures and attributes.

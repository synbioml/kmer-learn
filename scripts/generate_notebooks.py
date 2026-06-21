"""Generate all 13 example notebooks for kmer-learn.

Each notebook:
- Has a vignette index at the top (cross-links to all others)
- Is executed in-place (outputs preserved)
- Uses the new numbering after moving distance_kernel to position 02
"""

import json
import os
import sys
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "..", "examples")

VIGNETTES = [
    (1,  "basic_kernel_matrix",              "GKMKernel basics"),
    (2,  "distance_metrics_and_kernels",     "Distance metrics & kernels"),
    (3,  "svc_with_kernel",                  "SVM with kernel"),
    (4,  "clustering_sequences",             "Clustering"),
    (5,  "score_long_sequence",              "Long sequence scoring"),
    (6,  "weighted_kernel",                  "Weighted (WGKM) kernel"),
    (7,  "transform_and_comparison",         "Transforms & comparison"),
    (8,  "windowed_3d_tensors",              "Windowed 3D tensors"),
    (9,  "spectrum_encoder_and_differential", "Spectrum encoder & NB"),
    (10, "gappy_encoder",                    "Gappy encoder"),
    (11, "mismatch_encoder",                 "Mismatch encoder"),
    (12, "shuffler_and_chunker",             "Shuffler & chunker"),
]


def vignette_index(current):
    lines = ["**Vignette index:**"]
    for num, stem, title in VIGNETTES:
        if num == current:
            lines.append(f"`**{num:02d}**` {title}")
        else:
            lines.append(f"[`{num:02d}` {title}]({num:02d}_{stem}.ipynb)")
    return " | ".join(lines)


def make_notebook(cells_metadata):
    nb = new_notebook()
    for cell_type, source in cells_metadata:
        if cell_type == "md":
            nb.cells.append(new_markdown_cell(source))
        elif cell_type == "code":
            nb.cells.append(new_code_cell(source))
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return nb


def execute_and_save(nb, path, num):
    print(f"  Executing notebook {num:02d}...")
    client = NotebookClient(nb, timeout=120, kernel_name="python3",
                            resources={"metadata": {"path": os.path.dirname(path)}})
    try:
        client.execute()
    except Exception as e:
        print(f"  ERROR executing notebook {num:02d}: {e}", file=sys.stderr)
        raise
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved {path}")


def nb_01():
    idx = vignette_index(1)
    cells = [
        ("md", f"# 01 — GKMKernel: Basic Kernel Matrix\n\n{idx}\n\nThis vignette shows how to build a gkm kernel matrix and verify its basic invariants (symmetry, unit diagonal, PSD)."),
        ("code", """import numpy as np
from kmer.kernels import GKMKernel

seqs = [
    "ACGTACGTACGTACGTACGT",
    "TTTTAAAAGGGGCCCCAAAA",
    "ACGTTGCATGCATGCATGCA",
    "CCCCGGGGTTTTAAAACCCC",
    "ATATGCGCATATGCGCATAT",
    "GAATTCGAATTCGAATTCGA",
]

kern = GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True)
kern.set_references(seqs)
K = np.asarray(kern.kernel())
print("Shape:", K.shape)
print("Symmetric:", np.allclose(K, K.T))
print("Unit diagonal:", np.allclose(np.diag(K), 1.0))
print("Min eigenvalue:", np.linalg.eigvalsh(K).min(), "(PSD if >= 0)")"""),
        ("code", """# Cross-kernel: query sequences vs reference set
queries = ["ACGTACGTACGTACGTACGTAC", "CCCCGGGGTTTTAAAACCCCAG"]
kern.set_references(seqs)
Kq = np.asarray(kern.kernel(X_query=queries))
print("Cross-kernel shape:", Kq.shape)"""),
    ]
    return make_notebook(cells)


def nb_02():
    idx = vignette_index(2)
    cells = [
        ("md", f"# 02 — Distance Metrics & Distance Kernels\n\n{idx}\n\n`kmer.distance` provides pairwise sequence metrics and a `DistanceKernel` that turns any distance into a kernel.\n\n**Optional backends (strongly recommended for speed):**\n- `rapidfuzz` — 10-100x faster Hamming/Levenshtein.\n- `parasail` — SIMD-accelerated NW/SW alignment.\n\n```bash\npip install rapidfuzz parasail\n```\n\nAll metrics implement `BaseDistance` with `__call__`, `cdist`, and `pdist`. `DistanceKernel` wraps any `BaseDistance` with a post-transform (RBF, Laplacian, Cauchy, etc.) to produce a PSD Gram matrix suitable for `SVC(kernel='precomputed')`."),
        ("code", """import numpy as np
import matplotlib.pyplot as plt
from kmer.distance import Hamming, Levenshtein, NeedlemanWunsch, SmithWaterman, DistanceKernel

import importlib.util
print(f"rapidfuzz available: {importlib.util.find_spec('rapidfuzz') is not None}")
print(f"parasail available: {importlib.util.find_spec('parasail') is not None}")"""),
        ("md", "## Part I — Distance Metrics\n\n## 1. Hamming distance (equal-length only)"),
        ("code", """print('ACGT vs ACGA:', Hamming()('ACGT', 'ACGA'))
print('ACGT vs TGCA:', Hamming()('ACGT', 'TGCA'))"""),
        ("md", "## 2. Levenshtein distance"),
        ("code", """print('ACGT vs ACGTA (insertion):', Levenshtein()('ACGT', 'ACGTA'))
print('ACGT vs ACG  (deletion):', Levenshtein()('ACGT', 'ACG'))
print('ACGT vs ACGA (substitution):', Levenshtein()('ACGT', 'ACGA'))"""),
        ("md", "## 3. Pairwise distance matrix (heatmap)"),
        ("code", """seqs = ['ACGTACGT', 'ACGTACGA', 'TTTTAAAA', 'ACGTACGT', 'GGGGCCCC']
D = Levenshtein().cdist(seqs)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
im = ax.imshow(D, cmap='viridis')
ax.set_xticks(range(len(seqs)))
ax.set_yticks(range(len(seqs)))
ax.set_xticklabels([s[:6] for s in seqs], rotation=45, ha='right')
ax.set_yticklabels([s[:6] for s in seqs])
ax.set_title('Levenshtein distance matrix')
plt.colorbar(im, ax=ax, label='Distance')
plt.show()"""),
        ("md", "## 4. Needleman-Wunsch (global alignment score)"),
        ("code", """nw = NeedlemanWunsch(match=1, mismatch=-1, gap=-1)
print('ACGT vs ACGT:', nw('ACGT', 'ACGT'))
print('ACGT vs ACG:', nw('ACGT', 'ACG'))
print('ACGT vs AAAA:', nw('ACGT', 'AAAA'))"""),
        ("md", "## 5. Smith-Waterman (local alignment score)"),
        ("code", """sw = SmithWaterman(match=1, mismatch=-1, gap=-1)
print('ACGT vs TTACGTTT:', sw('ACGT', 'TTACGTTT'))"""),
        ("md", "## 6. Custom substitution matrix (NUC4.4)"),
        ("code", """nw = NeedlemanWunsch(substitution_matrix='NUC4.4', gap=-4)
print('ACGT vs ACGT (NUC4.4):', nw('ACGT', 'ACGT'))"""),
        ("md", "## 7. NW score matrix (heatmap)"),
        ("code", """seqs = ['ACGTACGT', 'ACGTACGA', 'ACGTTCGT', 'TTTTAAAA', 'GGGGCCCC']
nw = NeedlemanWunsch(match=2, mismatch=-1, gap=-1)
S = nw.cdist(seqs)

fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
im = ax.imshow(S, cmap='RdYlGn')
ax.set_xticks(range(len(seqs)))
ax.set_yticks(range(len(seqs)))
ax.set_xticklabels(seqs, rotation=45, ha='right')
ax.set_yticklabels(seqs)
ax.set_title('Needleman-Wunsch alignment scores')
plt.colorbar(im, ax=ax, label='Score')
plt.show()"""),
        ("md", "---\n\n## Part II — Distance Kernels\n\n`DistanceKernel` wraps any `BaseDistance` with a post-transform to produce a kernel matrix.\n\n## 8. RBF kernel on Levenshtein distance"),
        ("code", """seqs = ['ACGT', 'ACGTA', 'TTTT', 'ACGTACGT', 'GCGCGCGC']
kern = DistanceKernel(Levenshtein(), transform='rbf', gamma=0.1)
K = kern.kernel_matrix(seqs)
print("Gram matrix (rounded):")
print(np.round(K, 3))"""),
        ("md", "## 9. Verify PSD property\n\nRBF, Laplacian, and Cauchy transforms on proper metrics should produce PSD Gram matrices."),
        ("code", """for transform in ['rbf', 'laplacian', 'cauchy']:
    kern = DistanceKernel(Levenshtein(), transform=transform, gamma=0.1)
    K = kern.kernel_matrix(seqs)
    eigvals = np.linalg.eigvalsh(K)
    print(f"{transform:10s}: min eigenvalue = {eigvals.min():.6e} (PSD: {eigvals.min() >= -1e-10})")"""),
        ("md", "## 10. Available transforms"),
        ("code", """for transform in ['linear', 'rbf', 'poly', 'sigmoid', 'laplacian', 'exponential', 'exponential_clipped', 'cauchy']:
    kern = DistanceKernel(Hamming(), transform=transform, gamma=1.0)
    val = kern('ACGT', 'ACGA')
    print(f"{transform:22s}: K(ACGT, ACGA) = {val:.4f}")"""),
        ("md", "## 11. Train a KernelSVM\n\n**See also:** [03_svc_with_kernel.ipynb](03_svc_with_kernel.ipynb) for more on training SVMs with kernels."),
        ("code", """from kmer.models import KernelSVM
np.random.seed(0)
positives = ['ACGTACGT' + 'ACGT' * np.random.randint(0, 3) for _ in range(20)]
negatives = ['ATATATAT' + 'ATAT' * np.random.randint(0, 3) for _ in range(20)]
all_seqs = positives + negatives
y = np.array([1]*20 + [0]*20)

kern = DistanceKernel(Levenshtein(), transform='rbf', gamma=0.05)
clf = KernelSVM(kern, C=1.0)
clf.fit(all_seqs, y)
acc = (clf.predict(all_seqs) == y).mean()
print(f"Train accuracy: {acc:.2f}")"""),
    ]
    return make_notebook(cells)


def nb_03():
    idx = vignette_index(3)
    cells = [
        ("md", f"# 03 — SVM with a Precomputed Kernel\n\n{idx}\n\nTrain an SVM on a gkm kernel using the `KernelSVM` model. Compares three weight schemes (`full`, `estimated_full`, `truncated`) and the effect of `use_rc`."),
        ("code", """import numpy as np
from kmer.kernels import GKMKernel
from kmer.models import KernelSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(42)
def random_seq(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))

rng = np.random.default_rng(42)
positives = [("ACGTACGT" + random_seq(12, rng)) for _ in range(60)]
negatives = [random_seq(20, rng) for _ in range(60)]
seqs = positives + negatives
y = np.array([1]*60 + [0]*60)

seqs_train, seqs_test, y_train, y_test = train_test_split(
    seqs, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Train: {len(seqs_train)}, Test: {len(seqs_test)}")"""),
        ("md", "## Compare three weight schemes"),
        ("code", """schemes = ["full", "estimated_full", "truncated"]
results = {}
for scheme in schemes:
    kern = GKMKernel(L=10, k=6, d=3, kernel_type=scheme, use_rc=True)
    clf = KernelSVM(kern, C=1.0)
    clf.fit(seqs_train, y_train)
    preds = clf.predict(seqs_test)
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, clf.decision_function(seqs_test))
    results[scheme] = {"kern": kern, "clf": clf, "acc": acc, "auc": auc}
    print(f"{scheme:18s}: acc={acc:.3f}, AUC={auc:.3f}")"""),
        ("md", "## The effect of `use_rc`"),
        ("code", """# Build a dataset where half the positives have the RC motif
positives_rc = [("ACGTACGT"[::-1].translate(str.maketrans("ACGT", "TGCA")) + random_seq(12, rng))
                for _ in range(30)]
positives_fwd = [("ACGTACGT" + random_seq(12, rng)) for _ in range(30)]
positives_mix = positives_fwd + positives_rc
negatives = [random_seq(20, rng) for _ in range(60)]
seqs_mix = positives_mix + negatives
y_mix = np.array([1]*60 + [0]*60)
seqs_tr, seqs_te, y_tr, y_te = train_test_split(seqs_mix, y_mix, test_size=0.3, stratify=y_mix, random_state=42)

for use_rc in [True, False]:
    kern = GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=use_rc)
    clf = KernelSVM(kern, C=1.0)
    clf.fit(seqs_tr, y_tr)
    auc = roc_auc_score(y_te, clf.decision_function(seqs_te))
    print(f"use_rc={use_rc}: AUC={auc:.3f}")"""),
    ]
    return make_notebook(cells)


def nb_04():
    idx = vignette_index(4)
    cells = [
        ("md", f"# 04 — Hierarchical Clustering with Kernel Distances\n\n{idx}\n\nUse a gkm kernel to compute pairwise distances and cluster sequences hierarchically."),
        ("code", """import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from kmer.kernels import GKMKernel

np.random.seed(0)
def random_seq(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))
rng = np.random.default_rng(0)
group_a = ["ACGTACGT" + random_seq(12, rng) for _ in range(5)]
group_b = ["TTTTAAAA" + random_seq(12, rng) for _ in range(5)]
group_c = ["GCGCGCGC" + random_seq(12, rng) for _ in range(5)]
seqs = group_a + group_b + group_c
labels = ["A"]*5 + ["B"]*5 + ["C"]*5

kern = GKMKernel(L=8, k=5, d=2, kernel_type="truncated", use_rc=True)
kern.set_references(seqs)
K = np.asarray(kern.kernel())

# Distance: d = sqrt(2 - 2*K(a,b)) since diagonal is 1.0
D = np.sqrt(np.maximum(2 - 2*K, 0))
np.fill_diagonal(D, 0)
D = (D + D.T) / 2

condensed = squareform(D, checks=False)
Z = linkage(condensed, method="average")

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
dendrogram(Z, labels=labels, ax=ax)
ax.set_ylabel("Kernel distance")
plt.show()"""),
    ]
    return make_notebook(cells)


def nb_05():
    idx = vignette_index(5)
    cells = [
        ("md", f"# 05 — Scoring a Long Sequence with gkmSVM\n\n{idx}\n\nTrain a `KernelSVM` on short sequences, then scan a long sequence using the gkm kernel's `sliding_window_query_kernel` to find where the TF-binding signal is strongest."),
        ("code", """import numpy as np
import matplotlib.pyplot as plt
from kmer.kernels import GKMKernel
from kmer.models import KernelSVM

np.random.seed(0)
def random_seq(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))
rng = np.random.default_rng(0)

positives = ["ACGTACGT" + random_seq(12, rng) for _ in range(40)]
negatives = [random_seq(20, rng) for _ in range(40)]
seqs_train = positives + negatives
y_train = np.array([1]*40 + [0]*40)

kern = GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True)
clf = KernelSVM(kern, C=1.0)
clf.fit(seqs_train, y_train)
print("Trained.")"""),
        ("md", "## Build a long sequence with a motif at position 150"),
        ("code", """LONG_LEN = 300
MOTIF_POS = 150
rng2 = np.random.default_rng(99)
long_seq = list(random_seq(LONG_LEN, rng2))
for i, c in enumerate("ACGTACGTACGT"):
    long_seq[MOTIF_POS + i] = c
long_seq = "".join(long_seq)
print(f"Long sequence length: {len(long_seq)}")"""),
        ("md", "## Score the long sequence with a sliding window"),
        ("code", """sv_indices = clf._svc_.support_
sv_seqs = [clf._train_seqs_[i] for i in sv_indices]
sv_coef = clf._svc_.dual_coef_[0]
intercept = clf._svc_.intercept_[0]

kern_scan = GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True)
kern_scan.set_references(sv_seqs)

WINDOW = 20
SHIFT = 5
W = kern_scan.sliding_window_query_kernel([long_seq], window=WINDOW, shift=SHIFT)
K_windows = np.asarray(W[0])

scores = K_windows @ sv_coef + intercept

window_starts = np.arange(len(scores)) * SHIFT
fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
ax.plot(window_starts, scores, "-")
ax.axvline(MOTIF_POS, color="r", linestyle="--", label=f"Motif at {MOTIF_POS}")
ax.set_xlabel("Window start position")
ax.set_ylabel("SVM score")
ax.set_title("Sliding-window scan of long sequence")
ax.legend()
plt.show()

peak_pos = window_starts[np.argmax(scores)]
print(f"Peak at position {peak_pos} (motif at {MOTIF_POS})")"""),
    ]
    return make_notebook(cells)


def nb_06():
    idx = vignette_index(6)
    cells = [
        ("md", f"# 06 — Weighted (WGKM) Kernel\n\n{idx}\n\n`WGKMKernel` applies positional weighting to L-mers via a spatial kernel (triangular, Gaussian, etc.). When the motif is near the **center** of training sequences, WGKM should outperform the uniform GKM kernel."),
        ("code", """import numpy as np
from kmer.kernels import GKMKernel, WGKMKernel
from kmer.models import KernelSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
def random_seq(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))

SEQ_LEN = 30
MOTIF = "ACGTACGTACGT"
MOTIF_START = (SEQ_LEN - len(MOTIF)) // 2  # centered

rng = np.random.default_rng(42)
positives = []
for _ in range(60):
    s = list(random_seq(SEQ_LEN, rng))
    for i, c in enumerate(MOTIF):
        s[MOTIF_START + i] = c
    positives.append("".join(s))
negatives = [random_seq(SEQ_LEN, rng) for _ in range(60)]
seqs = positives + negatives
y = np.array([1]*60 + [0]*60)
seqs_tr, seqs_te, y_tr, y_te = train_test_split(seqs, y, test_size=0.3, stratify=y, random_state=42)
print(f"Train: {len(seqs_tr)}, Test: {len(seqs_te)}")"""),
        ("md", "## Compare GKM (uniform) vs WGKM (positional weighting)\n\nWith the motif centered, WGKM's positional weighting should boost performance."),
        ("code", """results = []
for name, kern in [
    ("GKM (uniform)", GKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True)),
    ("WGKM (Gaussian, sigma=5)", WGKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True,
                                              weight_kernel="gaussian", weight_sigma=5.0, weight_peak=1.0)),
    ("WGKM (Laplacian, gamma=0.1)", WGKMKernel(L=10, k=6, d=3, kernel_type="truncated", use_rc=True,
                                                 weight_kernel="laplacian", weight_gamma=0.1, weight_peak=1.0)),
]:
    clf = KernelSVM(kern, C=1.0)
    clf.fit(seqs_tr, y_tr)
    auc = roc_auc_score(y_te, clf.decision_function(seqs_te))
    results.append((name, auc))
    print(f"{name:35s}: AUC={auc:.3f}")"""),
    ]
    return make_notebook(cells)


def nb_07():
    idx = vignette_index(7)
    cells = [
        ("md", f"# 07 — Post-Transforms and Kernel Comparison\n\n{idx}\n\nVisualize the 6 post-transforms with sensible parameter values, then compare GKM vs WGKM across schemes and transforms on a **centered-motif** dataset where WGKM has an advantage."),
        ("code", """import numpy as np
import matplotlib.pyplot as plt
from kmer.kernels import GKMKernel, WGKMKernel
from kmer.models import KernelSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def random_seq(n, rng):
    return "".join(rng.choice(list("ACGT"), size=n))"""),
        ("md", "## 1. Visualize the transform kernels\n\nEach transform maps the normalized kernel value K ∈ [0, 1] to a new value. We use parameters that show each transform's shape clearly."),
        ("code", """def plot_transforms(gamma=2.0, coef0=1.0, degree=2, radius=0.1):
    K = np.linspace(0, 1, 200)
    transforms = {
        "linear":              K,
        "rbf":                 np.exp(gamma * (K - 1.0)),
        "poly":                (gamma * K + coef0) ** degree,
        "sigmoid":             np.tanh(gamma * K + coef0),
        "exponential":         np.exp(-gamma * (1.0 - K - radius)),
        "exponential_clipped": np.exp(-gamma * np.maximum(0.0, 1.0 - K - radius)),
    }
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for name, y in transforms.items():
        ax.plot(K, y, label=name)
    ax.set_xlabel("Normalized kernel K(a, b)")
    ax.set_ylabel("Transformed value")
    ax.set_title(f"Post-transforms (gamma={gamma}, coef0={coef0}, degree={degree}, radius={radius})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 5)
    plt.show()

plot_transforms(gamma=2.0, coef0=1.0, degree=2, radius=0.1)"""),
        ("md", "## 2. Centered-motif dataset"),
        ("code", """SEQ_LEN = 30
MOTIF = "ACGTACGTACGT"
MOTIF_START = (SEQ_LEN - len(MOTIF)) // 2

rng = np.random.default_rng(42)
positives = []
for _ in range(60):
    s = list(random_seq(SEQ_LEN, rng))
    for i, c in enumerate(MOTIF):
        s[MOTIF_START + i] = c
    positives.append("".join(s))
negatives = [random_seq(SEQ_LEN, rng) for _ in range(60)]
seqs = positives + negatives
y = np.array([1]*60 + [0]*60)
seqs_tr, seqs_te, y_tr, y_te = train_test_split(seqs, y, test_size=0.3, stratify=y, random_state=42)"""),
        ("md", "## 3. Compare schemes × transforms"),
        ("code", """schemes = ["full", "estimated_full", "truncated"]
transforms = ["linear", "rbf", "poly", "sigmoid", "exponential", "exponential_clipped"]

results = []
for kern_type in ["GKM", "WGKM"]:
    for scheme in schemes:
        for tf in transforms:
            if kern_type == "GKM":
                kern = GKMKernel(L=10, k=6, d=3, kernel_type=scheme, use_rc=True,
                                 transform=tf, transform_gamma=2.0, transform_coef0=1.0,
                                 transform_degree=2, transform_radius=0.1)
            else:
                kern = WGKMKernel(L=10, k=6, d=3, kernel_type=scheme, use_rc=True,
                                  weight_kernel="gaussian", weight_sigma=5.0, weight_peak=1.0,
                                  transform=tf, transform_gamma=2.0, transform_coef0=1.0,
                                  transform_degree=2, transform_radius=0.1)
            clf = KernelSVM(kern, C=1.0)
            clf.fit(seqs_tr, y_tr)
            auc = roc_auc_score(y_te, clf.decision_function(seqs_te))
            results.append((kern_type, scheme, tf, auc))

print(f"{'kernel':<5} {'scheme':<18} {'transform':<22} {'AUC':>6}")
print("-" * 55)
for r in results:
    print(f"{r[0]:<5} {r[1]:<18} {r[2]:<22} {r[3]:.3f}")"""),
        ("code", """# Line plot of AUC by transform, grouped by kernel type
fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
for kern_type in ["GKM", "WGKM"]:
    aucs = [r[3] for r in results if r[0] == kern_type and r[1] == "truncated"]
    ax.plot(transforms, aucs, marker="o", label=f"{kern_type} (truncated)")
ax.set_xlabel("Transform")
ax.set_ylabel("Test AUC")
ax.set_title("GKM vs WGKM (truncated scheme, centered motif)")
ax.legend()
ax.set_ylim(0.4, 1.05)
plt.show()"""),
    ]
    return make_notebook(cells)


def nb_08():
    idx = vignette_index(8)
    cells = [
        ("md", f"# 08 — Windowed 3D Tensors\n\n{idx}\n\n`WindowedGKMKernel` produces a 3D tensor of shape `(n_windows, n_sequences, n_sequences)`. By tracking the kernel value at each window position, we can see how sequence similarity changes along the sequences."),
        ("code", """import numpy as np
import matplotlib.pyplot as plt
from kmer.kernels import WindowedGKMKernel

seq_a = "ACGTACGTACGT" + "TTTTGGGGCCCC"  # motif at start
seq_b = "TTTTGGGGCCCC" + "ACGTACGTACGT"  # motif at end
seqs = [seq_a, seq_b]

kern = WindowedGKMKernel(L=6, k=4, d=2, window=8, shift=2, padding=0)
kern.set_references(seqs)
T = np.asarray(kern.kernel())
print("Tensor shape:", T.shape)"""),
        ("md", "## Line plot of kernel value per window\n\nThe diagonal (self-kernel) is constant at 1.0. The off-diagonal (cross-kernel) peaks where the two sequences share a motif in the same window."),
        ("code", """n_windows = T.shape[0]
window_starts = np.arange(n_windows) * 2

fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
ax.plot(window_starts, T[:, 0, 0], "b-", label="K(seq_a, seq_a) [self]")
ax.plot(window_starts, T[:, 1, 1], "g-", label="K(seq_b, seq_b) [self]")
ax.plot(window_starts, T[:, 0, 1], "r-", label="K(seq_a, seq_b) [cross]")
ax.set_xlabel("Window start position")
ax.set_ylabel("Kernel value")
ax.set_title("Windowed kernel: similarity vs position")
ax.legend()
ax.set_ylim(0, 1.1)
plt.show()

peak = window_starts[np.argmax(T[:, 0, 1])]
print(f"Peak cross-kernel at window position {peak}")"""),
    ]
    return make_notebook(cells)


def nb_09():
    idx = vignette_index(9)
    cells = [
        ("md", f"# 09 — Spectrum Encoder & Differential k-mer Scoring\n\n{idx}\n\nThis vignette demonstrates:\n1. Using `SpectrumEncoder` to convert sequences into a sparse CSR matrix of k-mer counts.\n2. Using `DifferentialKmerScorer` (Multinomial Naive Bayes) to identify enriched k-mers, with auto-generated negatives via `KmerShuffler`.\n\n**See also:** [12_shuffler_and_chunker.ipynb](12_shuffler_and_chunker.ipynb) for background-model details."),
        ("code", """import numpy as np
from kmer.encoders import SpectrumEncoder
from kmer.models import DifferentialKmerScorer
from kmer.perturb import KmerShuffler"""),
        ("md", "## 1. Encode sequences with SpectrumEncoder"),
        ("code", """seqs = ['ACGTACGTACGT', 'TTTTAAAACCCC', 'GCGCGCGCGCGC']
enc = SpectrumEncoder(k=4, canonical_rc=True)
X = enc.fit_transform(seqs)
print('Shape:', X.shape)
print('Format:', X.format)
print('Feature names[:5]:', enc.get_feature_names_out()[:5])
print('Total counts per seq:', X.sum(axis=1).A1)"""),
        ("md", "## 2. Differential k-mer scoring"),
        ("code", """np.random.seed(0)
positives = ['ACGTACGT' + 'ACGT' * np.random.randint(0, 3) for _ in range(50)]

scorer = DifferentialKmerScorer(
    featurizer=SpectrumEncoder(k=4),
    background=KmerShuffler(k=1, seed=42),
)
scorer.fit(positives)

print('Top 5 enriched k-mers:')
print(scorer.kmer_scores_.sort_values(ascending=False).head(5))"""),
        ("code", """test_seqs = ['ACGTACGTACGTACGT', 'TTTTAAAAGGGGCCCC', 'GCGCGCGCGCGCGCGC']
scores = scorer.decision_function(test_seqs)
for s, sc in zip(test_seqs, scores):
    print(f'  {s}: {sc:.3f}')"""),
        ("md", "## 3. Effect of background stringency"),
        ("code", """for bg_k in [1, 2]:
    s = DifferentialKmerScorer(
        featurizer=SpectrumEncoder(k=4),
        background=KmerShuffler(k=bg_k, seed=42),
    )
    s.fit(positives)
    top = s.kmer_scores_.sort_values(ascending=False).head(3)
    print(f'background k={bg_k}: top 3 = {top.to_dict()}')"""),
    ]
    return make_notebook(cells)


def nb_10():
    idx = vignette_index(10)
    cells = [
        ("md", f"# 10 — Gappy Encoder\n\n{idx}\n\nThe `GappyEncoder` counts gappy k-mers — patterns with don't-care positions, e.g. `*--*` matches `AC.G`, `GT.A`, etc.\n\nTwo construction modes:\n- **Explicit masks:** `masks=[\"*--*\", \"*-*--*\"]`\n- **Gap range:** `L=6, g_min=2, g_max=3`"),
        ("code", """from kmer.encoders import GappyEncoder
import numpy as np"""),
        ("md", "## 1. Explicit mask with palindromic k-mer demonstration\n\nThe sequence `CGGCCGGCG` with mask `*--*` (concrete positions 0 and 3) produces gappy k-mers: `C--G`, `G--G`, `G--C`, `C--C`, `G--G`, `C--C`, `G--G`. With `canonical_rc=True`, these collapse: `C--G` and `G--C` are RC pairs; `G--G` and `C--C` are RC pairs; `G--G` is its own RC (palindrome), so it gets counted twice."),
        ("code", """seq = 'CGGCCGGCG'
enc = GappyEncoder(masks=['*--*'])
X = enc.fit_transform([seq])
print('Without canonical_rc:')
print('  Shape:', X.shape)
print('  Total count:', X.sum())

enc_rc = GappyEncoder(masks=['*--*'], canonical_rc=True)
X_rc = enc_rc.fit_transform([seq])
print('With canonical_rc=True:')
print('  Shape:', X_rc.shape)
print('  Total count:', X_rc.sum())
print('  Non-zero columns:', X_rc.nnz)"""),
        ("md", "## 2. Multiple masks"),
        ("code", """enc = GappyEncoder(masks=['*--*', '*---*', '*-*--*'])
X = enc.fit_transform(['ACGTACGTACGT'])
print('Shape:', X.shape)
print('Masks:', enc.masks_)"""),
        ("md", "## 3. Gap range"),
        ("code", """enc = GappyEncoder(L=5, g_min=2, g_max=3)
X = enc.fit_transform(['ACGTACGTACGT'])
print(f'n_masks: {len(enc.masks_)} (C(5,2)+C(5,3) = {10+10})')
print('Masks:', enc.masks_)
print('Shape:', X.shape)"""),
    ]
    return make_notebook(cells)


def nb_11():
    idx = vignette_index(11)
    cells = [
        ("md", f"# 11 — Mismatch Encoder\n\n{idx}\n\nThe `MismatchEncoder` (Leslie, Eskin, Noble 2004) counts k-mers with up to `m` mismatches. For each k-mer in the sequence, it increments counts for ALL k-mers within Hamming distance `m`.\n\n- `m=0`: equivalent to `SpectrumEncoder`.\n- `m=1`: each k-mer contributes to 1 + k×3 neighbors.\n- `m=2`: each k-mer contributes to 1 + k×3 + C(k,2)×9 neighbors."),
        ("code", """import numpy as np
from kmer.encoders import MismatchEncoder, SpectrumEncoder"""),
        ("md", "## 1. Basic usage"),
        ("code", """enc = MismatchEncoder(k=4, m=1)
X = enc.fit_transform(['ACGTACGT'])
print('Shape:', X.shape)
print('Total counts:', X.sum())"""),
        ("md", "## 2. m=0 matches SpectrumEncoder"),
        ("code", """enc_m0 = MismatchEncoder(k=4, m=0)
enc_sp = SpectrumEncoder(k=4)
X_m0 = enc_m0.fit_transform(['ACGTACGT'])
X_sp = enc_sp.fit_transform(['ACGTACGT'])
print('m=0 matches spectrum:', np.allclose(X_m0.toarray(), X_sp.toarray()))"""),
        ("md", "## 3. Effect of m on feature density"),
        ("code", """seqs = ['ACGTACGTACGTACGT']
for m in [0, 1, 2]:
    enc = MismatchEncoder(k=6, m=m)
    X = enc.fit_transform(seqs)
    print(f'm={m}: nnz = {X.nnz}, total = {X.sum()}')"""),
    ]
    return make_notebook(cells)


def nb_12():
    idx = vignette_index(12)
    cells = [
        ("md", f"# 12 — Shuffler & Chunker for Negative-Set Generation\n\n{idx}\n\n`kmer.perturb` provides two background models for generating null-model sequences:\n\n- **`KmerShuffler`** — preserves exact k-mer composition via random Eulerian paths in the De Bruijn graph.\n- **`Chunker`** — block-level perturbation: split into chunks, optionally RC, shuffle, concatenate.\n\nBoth implement `BaseBackgroundModel` with `perturb()` / `perturb_many()`."),
        ("code", """from collections import Counter
from kmer.perturb import KmerShuffler, Chunker, BaseBackgroundModel"""),
        ("md", "## 1. KmerShuffler — k-mer composition preservation"),
        ("code", """seq = 'AAACCCGGGTTTAAACCCGGGTTT'
for k in [1, 2, 3]:
    sh = KmerShuffler(k=k, seed=42)
    out = sh.shuffle(seq)
    in_kmers = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    out_kmers = Counter(out[i:i+k] for i in range(len(out) - k + 1))
    print(f'k={k}: in==out? {in_kmers == out_kmers}, out={out}')"""),
        ("md", "## 2. Endpoint modes"),
        ("code", """seq = 'ACGTACGTACGTAAACCCGGGTTT'
for mode in ['preserve', 'free', 'crop']:
    sh = KmerShuffler(k=3, seed=42, endpoints=mode)
    out = sh.shuffle(seq)
    print(f'{mode:8s}: len={len(out):2d} out={out}')"""),
        ("md", "## 3. Chunker — block-level perturbation"),
        ("code", """ch = Chunker(min_size=2, max_size=4, seed=42)
seq = 'ACGTACGTACGTACGT'
out = ch.chunk(seq)
print(f'in : {seq}')
print(f'out: {out}')
print(f'len preserved: {len(out) == len(seq)}')
print(f'composition preserved: {Counter(seq) == Counter(out)}')"""),
        ("md", "## 4. Chunker with strand flipping"),
        ("code", """ch = Chunker(min_size=2, max_size=4, flip_strand=True, seed=42)
out = ch.chunk('ACGTACGTACGTACGT')
print(f'with flip_strand: {out}')"""),
        ("md", "## 5. Batch operations (parallel, reproducible)"),
        ("code", """seqs = ['ACGTACGTACGT', 'TTTTAAAACCCC', 'GCGCGCGCGCGC']
r1 = KmerShuffler(k=2, seed=42).shuffle_many(seqs, n_jobs=1)
r2 = KmerShuffler(k=2, seed=42).shuffle_many(seqs, n_jobs=4)
print(f'n_jobs=1 == n_jobs=4: {r1 == r2}')"""),
    ]
    return make_notebook(cells)




def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)
    for f in os.listdir(EXAMPLES_DIR):
        if f.endswith(".ipynb"):
            os.remove(os.path.join(EXAMPLES_DIR, f))

    notebooks = [
        (1,  "basic_kernel_matrix",              nb_01),
        (2,  "distance_metrics_and_kernels",     nb_02),
        (3,  "svc_with_kernel",                  nb_03),
        (4,  "clustering_sequences",             nb_04),
        (5,  "score_long_sequence",              nb_05),
        (6,  "weighted_kernel",                  nb_06),
        (7,  "transform_and_comparison",         nb_07),
        (8,  "windowed_3d_tensors",              nb_08),
        (9,  "spectrum_encoder_and_differential", nb_09),
        (10, "gappy_encoder",                    nb_10),
        (11, "mismatch_encoder",                 nb_11),
        (12, "shuffler_and_chunker",             nb_12),
    ]
    for num, stem, builder in notebooks:
        path = os.path.join(EXAMPLES_DIR, f"{num:02d}_{stem}.ipynb")
        print(f"Building {path}...")
        nb = builder()
        execute_and_save(nb, path, num)
    print("All notebooks generated.")


if __name__ == "__main__":
    main()

# SCOUT: OOD Detection via kNN + Finding-Vocabulary Fusion

SCOUT scores each image with `max(z_knn, z_vocab)`:

- **z_knn** — z-scored k-nearest-neighbor distance (k=20) of the image embedding
  against an in-distribution (ID) dev set, in the vision encoder's embedding space.
- **z_vocab** — z-scored "finding-vocabulary" score: the mean cosine similarity of
  the image embedding to a set of **unseen/OOD finding terms**, minus its mean
  similarity to a set of **known/ID finding terms** (both encoded with the same
  model's text tower).

The fusion score is evaluated as an OOD detector (ID vs. semantic-shift /
covariate-shift / far-OOD) with AUROC, AUPR, and FPR@95%TPR, averaged over 5
random dev/eval splits (seeds `42, 1, 7, 123, 2024`), plus a paired bootstrap
(1000 resamples) comparing the fusion score against each raw component.

Two scripts, two vocabularies, same fusion rule and protocol:

| Script | Vocabulary | Output |
|---|---|---|
| `run_scout_final_with_bootstrap.py` | **Oracle** — unseen terms hand-picked to be clinically related to each dataset's true OOD pathology | `outputs/scout_final_verified_results.json` |
| `run_scout_generic_vocab.py` | **Generic** — class-agnostic, non-overlapping unseen terms (ablation: does SCOUT still work without an oracle vocabulary?) | `outputs/scout_generic_vocab_results.json` |

Both scripts run on two benchmarks per invocation: **ophthalmology** (fundus
photos, FLAIR encoder) and **dermatology** (dermoscopy images, BiomedCLIP
encoder).

---

## Quick start

```bash
# 1. Clone and set up environment
git clone https://github.com/Utathyaworks/Medcheck.git
cd Medcheck
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get FLAIR (needed for the ophthalmology benchmark)
git clone https://github.com/jusiro/FLAIR.git flair_repo
# then download flair_resnet.pth and place it at:
#   flair/modeling/flair_pretrained_weights/flair_resnet.pth
# (BiomedCLIP for dermatology downloads automatically on first run)

# 3. Download datasets (FIVES, DeepDRiD, ISIC 2019, PAD-UFES-20, ImageNet-1k val
#    — links in section 3 below) and build two manifest CSVs pointing at them:
#      data/manifests/benchmark_manifest.csv
#      data/manifests/dermatology_manifest.csv

# 4. Edit ROOT and the flair_repo path at the top of both scripts to match
#    your layout:
#      ROOT = Path("/workspace/medical_ood")
#      sys.path.insert(0, "/workspace/medical_ood/flair_repo")

# 5. Run
python run_scout_final_with_bootstrap.py     # oracle vocabulary
python run_scout_generic_vocab.py            # generic vocabulary ablation
```

Results are written to `outputs/scout_final_verified_results.json` and
`outputs/scout_generic_vocab_results.json`, with progress printed live.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

GPU with CUDA is strongly recommended (embeds ~1,500 images per benchmark
through a ResNet/ViT encoder); CPU will work but is slow. The code auto-detects
`cuda` and falls back to `cpu`.

## 2. Get FLAIR (ophthalmology vision-language model)

The ophthalmology benchmark uses [FLAIR](https://github.com/jusiro/FLAIR), a
fundus vision-language model, with its pretrained checkpoint.

```bash
git clone https://github.com/jusiro/FLAIR.git flair_repo
```

Download the FLAIR pretrained checkpoint (`flair_resnet.pth`) per the FLAIR
repo's instructions and place it at:

```
<ROOT>/flair/modeling/flair_pretrained_weights/flair_resnet.pth
```

Dermatology uses `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
from the HuggingFace Hub via `open_clip` — it downloads automatically on first
run (requires internet access / HF cache).

## 3. Datasets

Both scripts read a CSV **manifest** that lists local file paths and group
labels — they do not download data themselves. Build the manifests from the
following public sources:

### Ophthalmology (`data/manifests/benchmark_manifest.csv`)

Columns: `sample_id, filepath_relative, source_dataset, split, group, class_label, source_filename, sha256, selected_for_eval, exclusion_reason`

| Group | Source dataset | N | Link |
|---|---|---|---|
| `id` | FIVES (Fundus Image Vessel Segmentation) | 300 | https://www.kaggle.com/datasets/khoongweihao/fives-a-fundus-image-dataset |
| `semantic` | FIVES (disease-positive subset) | 300 | same as above |
| `covariate` | DeepDRiD | 150 | https://github.com/deepdrdoi/DeepDRiD |
| `far_ood` | ImageNet-1k (validation split) | 150 | https://image-net.org/download.php |

### Dermatology (`data/manifests/dermatology_manifest.csv`)

Columns: `filepath, group, class_label`

| Group | Source dataset | N | Link |
|---|---|---|---|
| `id` | ISIC 2019 (NV — melanocytic nevus) | 300 | https://challenge.isic-archive.com/data/#2019 |
| `semantic` | ISIC 2019 (MEL — melanoma) | 300 | same as above |
| `covariate` | PAD-UFES-20 | 300 | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| `far_ood` | ImageNet-1k (validation split) | 300 | https://image-net.org/download.php |

Each manifest row's `filepath` / `filepath_relative` must point to an actual
image file on disk. Regenerate these CSVs with your own paths after
downloading the datasets above (`group` and `class_label` columns define the
ID/semantic/covariate/far_ood split used by the eval code — keep those
assignments, only the paths need to change).

## 4. Directory layout expected by the scripts

```
<ROOT>/
├── flair_repo/                     # cloned FLAIR repo (step 2)
├── flair/modeling/flair_pretrained_weights/flair_resnet.pth
├── data/
│   └── manifests/
│       ├── benchmark_manifest.csv       # ophthalmology
│       └── dermatology_manifest.csv     # dermatology
└── outputs/                        # created automatically
```

`ROOT` is hardcoded in both scripts as `/workspace/medical_ood`
(`ROOT = Path("/workspace/medical_ood")` near the top of each file) — change
this constant (and the `sys.path.insert(...)` line pointing at `flair_repo`)
to match your own layout before running.

## 5. Run

```bash
cd <ROOT>
python run_scout_final_with_bootstrap.py     # oracle vocabulary
python run_scout_generic_vocab.py            # generic vocabulary ablation
```

Each run prints per-seed progress and a final summary table to stdout, and
writes full per-seed metrics (+ paired bootstrap deltas, for the oracle script)
to the corresponding JSON file in `outputs/`.

## 6. Evaluation protocol (exact seeds, splits, vocab sizes)

- **Seeds:** `[42, 1, 7, 123, 2024]` — all metrics are computed once per seed,
  then reported as mean ± std across these 5 seeds.
- **Dev/eval split:** per group, per seed, `split_dev_eval(n, frac_dev=0.3, seed=seed)`
  shuffles indices with `np.random.RandomState(seed)` and takes the first 30%
  as the **dev** set (used only to build the kNN reference set and the
  z-score mean/std for both `knn` and `vocab`), the remaining 70% as the
  **eval** set (used for AUROC/AUPR/FPR95).
- **kNN:** `k=20` nearest neighbors, cosine similarity (`X_query @ X_dev.T`),
  against the ID dev embeddings only.
- **Paired bootstrap** (oracle script only): `n_boot=1000` resamples per
  `(seed, shift, comparison)`, same `seed` as the split, reports
  `delta_AUROC_pp` (fusion − component, in percentage points) and a 95% CI
  (2.5th/97.5th percentile of the bootstrap deltas).
- **Vocabulary size:** 10 "known/ID" terms + 10 "unseen/OOD" terms per
  dataset, for both the oracle and generic vocabularies (so 4 lists of 10
  terms total: ophthalmology known/unseen, dermatology known/unseen).

### Dataset counts (exact)

| Benchmark | Group | Count | Total |
|---|---|---|---|
| Ophthalmology | `id` | 300 | **900** |
| Ophthalmology | `semantic` | 300 | |
| Ophthalmology | `covariate` | 150 | |
| Ophthalmology | `far_ood` | 150 | |
| Dermatology | `id` | 300 | **1,200** |
| Dermatology | `semantic` | 300 | |
| Dermatology | `covariate` | 300 | |
| Dermatology | `far_ood` | 300 | |

Within each seed's 30/70 dev/eval split, these counts divide accordingly
(e.g. ophthalmology `id`: 90 dev / 210 eval; `covariate`: 45 dev / 105 eval).

## 7. Reference results

Precomputed outputs from a prior run are included in `outputs/` for reference:

- `scout_final_verified_results.json` / `scout_final_verified_log.txt` — oracle vocabulary
- `scout_generic_vocab_results.json` / `scout_generic_log.txt` — generic vocabulary

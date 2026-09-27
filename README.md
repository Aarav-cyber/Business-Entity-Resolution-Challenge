# Business Entity Resolution Challenge

An end-to-end scalable ML pipeline for multi-source **Business Entity Resolution**, built for the Amazon ML Challenge 2026.

**Team:** Aarav · Ankush · Riju · Anjali

---

## 📌 Problem Overview

Given noisy business entity records across **3 sources** (Source 1 = clean reference list; Source 2 & Source 3 = noisy datasets), the objective is to accurately identify all matching records in Source 2 and Source 3 for every Source 1 business entity.

### Key Challenge Constraints
- **Precision-heavy Metric ($F_{0.5}$, $\beta = 0.5$):** False merges are penalized **2× harder** than missed matches. Singletons (businesses with no matches in S2/S3) correctly predicted as empty score **1.0**, whereas false merges score **0.0**.
- **Graded Blocking Quality:** In addition to prediction accuracy (`matching_results.tsv`), candidate set size and reduction ratio (`candidate_pairs.tsv`) count toward final team ranking.
- **Open-Set Country Support:** Training data covers `{US, India}`, while test data contains unseen countries such as **France**. No country names may be hardcoded.
- **Fair Play & Model Constraints:** No external lookups (no geocoding APIs, no business registries, no internet augmentation). Models must be MIT/Apache-2.0 licensed with $\le 8\text{B}$ parameters.

---

## 🏗️ Pipeline Architecture

```text
train_source{1,2,3}.tsv, test_source{1,2,3}.tsv
         │
         ▼
[STAGE A] Data Loading + Normalization       (Owner: Aarav)
   - Unicode NFKC cleaning, landmark extraction, trailing legal suffix handling, open-set country normalization
         │
         ▼
[STAGE B] Blocking / Candidate Generation    (Owner: Ankush)
   - Country partitioning, inverted index, TF-IDF / fuzzy LSH candidate generator
   - Produces output/candidate_pairs.tsv
         │
         ▼
[STAGE C] Feature Engineering & Model        (Owner: Riju)
   - Pairwise name (char-bigram Dice + token Jaccard) and address
     (boilerplate-stripped bigram Dice) similarity, combined into one
     score: 0.55*name_dice + 0.25*name_jaccard + 0.20*addr_dice
   - Single threshold calibrated on train_ground_truth.tsv (maximizes
     F0.5 on positive/sampled-negative pairs) instead of a learned
     classifier - too few ground-truth pairs survive the train
     S1/S2/S3 sampling mismatch to train one reliably (see
     Documentation_template.md)
   - Produces output/matching_results.tsv, run as:
       python src/model/train.py --data-dir dataset/train
       python src/model/infer.py --data-dir dataset/test
         │
         ▼
[STAGE D] Evaluation & Submission Packaging (Owner: Anjali)
   - Macro F0.5 evaluation, validation split management, validate_submission.py format audit
   - Assembles final submission package
```

---

## 📁 Repository Structure

```text
business_entity_resolution/
├── dataset/
│   ├── train/                     # Raw training files (train_source1/2/3.tsv, train_ground_truth.tsv)
│   └── test/                      # Raw test files (test_source1/2/3.tsv)
├── output/
│   ├── matching_results.tsv       # Final predictions (Source 1 -> Matched S2/S3 IDs)
│   └── candidate_pairs.tsv        # Candidate set considered during blocking
├── src/
│   ├── data/                      # Stage A (Owner: Aarav)
│   │   ├── loader.py              # Robust TSV dataset loader & chunked streaming
│   │   ├── normalize.py           # Text/Name/Address/Country normalization engine
│   │   ├── eda.ipynb              # Dataset exploratory analysis
│   │   └── sample_cleaned.tsv     # Stage A schema handoff sample
│   ├── blocking/                  # Stage B (Owner: Ankush)
│   │   ├── blocking.py            # Layered candidate generation algorithms
│   │   └── run_blocking.py        # CLI entrypoint for blocking stage
│   ├── model/                     # Stage C (Owner: Riju)
│   │   ├── features.py            # Pairwise feature extraction
│   │   ├── train.py               # Model training & threshold optimization
│   │   └── infer.py               # Matching prediction pipeline
│   └── eval/                      # Stage D (Owner: Anjali)
│       ├── evaluate.py            # Local F0.5 & blocking metrics calculator
│       └── split.py               # Reproducible stratified validation split
├── tests/
│   └── test_stage_a.py            # Stage A normalization & loader unit tests
├── utils/
│   └── validate_submission.py     # Official format verification script
├── CONTRACTS.md                   # Frozen schema contracts between pipeline stages
├── requirements.txt               # Dependencies
└── README.md
```

---

## 👥 Stage Responsibilities & Ownership

| Stage | Owner | Responsibilities | Key Outputs |
|---|---|---|---|
| **Stage A: Data Foundation** | **Aarav** | TSV dataset loading, Unicode NFKC cleaning, trailing legal suffix stripping, landmark extraction, open-set country processing, dataset schema contracts. | `loader.py`, `normalize.py`, `sample_cleaned.tsv`, `CONTRACTS.md` |
| **Stage B: Blocking** | **Ankush** | Scalable candidate generation, inverted token indexing, TF-IDF cosine / fuzzy LSH ANN retrieval, candidate set tuning vs recall ceiling. | `candidate_pairs.tsv`, `run_blocking.py` |
| **Stage C: Matching Model** | **Riju** | Pairwise similarity feature engineering, gradient boosting model (LightGBM/XGBoost), $F_{0.5}$ precision-threshold tuning, singleton handling. | `matching_results.tsv`, `features.py`, `infer.py` |
| **Stage D: Eval & Packaging** | **Anjali** | Stratified validation split, macro $F_{0.5}$ evaluation, blocking quality metrics, `validate_submission.py` format validation, submission zip assembly. | `evaluate.py`, `split.py`, submission zip |

---

## 🚀 Quickstart & Execution

### 1. Environment Setup

```bash
git clone https://github.com/Aarav-cyber/Business-Entity-Resolution-Challenge.git
cd business_entity_resolution
pip install -r requirements.txt
```

### 2. Run Stage A Unit Tests

```bash
python -m pytest tests/test_stage_a.py
```

### 3. Pipeline Handoff Usage (Python API)

```python
from src.data.loader import load_source
from src.data.normalize import normalize_source

# Load raw source dataset
df_raw = load_source("dataset/train/train_source1.tsv")

# Run full Stage A normalization
df_clean = normalize_source(df_raw)

print(df_clean[["entity_id", "business_name_clean", "business_address_clean", "landmark", "country"]].head())
```

### 4. End-to-End Execution (Colab / Local)

```bash
# 1. Run Blocking (Stage B)
python src/blocking/run_blocking.py --data-dir dataset/test --out output/candidate_pairs.tsv

# (Optional Diagnostic) Evaluate Blocking Recall Ceiling against Ground Truth
python src/blocking/evaluate_blocking.py --candidates output/candidate_pairs.tsv --ground-truth dataset/train/train_ground_truth.tsv

# 2. Run Inference (Stage C)
python src/model/infer.py --data-dir dataset/test --candidates output/candidate_pairs.tsv --out output/matching_results.tsv

# 3. Validate Submission Format (Stage D)
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

---

## 📄 License & Compliance

- **Dependencies & Models:** Permissively licensed (MIT / Apache-2.0).
- **External Services:** 100% compliant with challenge fair play rules — zero external API lookups or internet augmentation used.

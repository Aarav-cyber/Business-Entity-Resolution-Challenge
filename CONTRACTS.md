# CONTRACTS.md — Frozen Data Interfaces

This document defines the frozen data schemas passed between pipeline stages.

---

## Stage A: Data Loading & Normalization (`src/data/`)
**Owner:** Aarav

### Output DataFrame / Cleaned TSV Schema:
| Column | Type | Description |
|---|---|---|
| `entity_id` | `str` | Unique record identifier (e.g. `S1-00001`, `S2-00047`, `S3-00812`) |
| `business_name` | `str` | Raw business name as loaded from source |
| `business_address` | `str` | Raw business address as loaded from source |
| `country` | `str` | Normalized country name (casing/whitespace cleaned; dynamic open set including France) |
| `business_name_clean` | `str` | Cleaned name: lowercased, stripped punctuation, normalized abbreviations, legal suffixes removed |
| `business_name_clean_with_suffix` | `str` | Cleaned name: lowercased, stripped punctuation, normalized abbreviations, legal suffixes retained |
| `business_address_clean` | `str` | Cleaned address: expanded Rd/St abbreviations, landmark phrases removed |
| `landmark` | `str` | Extracted landmark phrase (e.g. `"Near City Center"`, empty string `""` if none) |

*Rule:* Raw columns (`business_name`, `business_address`) must be preserved alongside `*_clean` columns; never overwrite raw values destructively.

---

## Stage B: Blocking / Candidate Generation (`src/blocking/`)
**Owner:** Ankush

- **Input:** Stage A cleaned DataFrames
- **Output File:** `output/candidate_pairs.tsv`
- **Format:** Tab-separated TSV
  ```tsv
  source1_entity_id<TAB>candidate_entity_ids
  ```
  - `candidate_entity_ids`: Comma-separated list of candidate S2 and S3 entity IDs (e.g. `S2-00047,S2-00193,S3-00812`).
  - Exactly **one row per Source 1 entity**.
  - No duplicates, S2-/S3- IDs only, empty string if zero candidates generated.

---

## Stage C: Feature Engineering & Matching Model (`src/model/`)
**Owner:** Riju

- **Input:** Stage A cleaned DataFrames + Stage B `output/candidate_pairs.tsv`
- **Output File:** `output/matching_results.tsv`
- **Format:** Tab-separated TSV
  ```tsv
  source1_entity_id<TAB>matched_entity_ids
  ```
  - `matched_entity_ids`: Comma-separated list of matched S2 and S3 entity IDs.
  - **Must be a strict subset** of `candidate_entity_ids` for that Source 1 entity.
  - Exactly **one row per Source 1 entity**. Singletons (no match) output an empty string for `matched_entity_ids`.

---

## Stage D: Evaluation & Submission Packaging (`src/eval/`)
**Owner:** Anjali

- **Input:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`, ground truth TSV
- **Output:**
  - Macro F₀.₅ score computation (precision weighted 2×).
  - Blocking metrics: Recall ceiling, Average candidate set size, Reduction ratio.
  - Verification pass from `utils/validate_submission.py`.

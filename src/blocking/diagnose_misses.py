"""
src/blocking/diagnose_misses.py
Author: Stage B Diagnostic Utility

Analyzes missed matches from evaluate_blocking to determine why true pairs
were discarded by Stage B candidate generation. Categorizes failure modes into:
  - Country partition mismatch (hard partition cut)
  - Missing/unnormalized country
  - Zero word-token overlap with high character n-gram similarity (typos/spelling noise)
  - Zero word-token overlap with low character similarity (aliases/acronyms/trade names)
  - High-frequency token cap cutoff (token fan-out cap >100 or max_token_df)
  - Address overlap without name match

Usage:
  python src/blocking/diagnose_misses.py \
      --candidates output/candidate_pairs.tsv \
      --ground-truth dataset/train/train_ground_truth.tsv \
      --data-dir dataset/train \
      --out output/missed_pairs_diagnostic.tsv
"""

import argparse
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Add repo root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_all_sources
from src.data.normalize import normalize_source


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from text."""
    s = f" {str(text).strip().lower()} "
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def get_word_tokens(text: str) -> Set[str]:
    """Extract clean word tokens of length > 1."""
    if not text or pd.isna(text):
        return set()
    return {t for t in re.findall(r"\b\w+\b", str(text).lower()) if len(t) > 1}


def parse_ground_truth(gt_path: str) -> Dict[str, Set[str]]:
    """Loads ground truth file mapping source1_entity_id -> set of matched entity IDs."""
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        if matched_str:
            cands = {c.strip() for c in matched_str.split(",") if c.strip()}
            gt_map[s1_id] = cands
        else:
            gt_map[s1_id] = set()
    return gt_map


def parse_candidates(cand_path: str) -> Dict[str, Set[str]]:
    """Loads candidate pairs file mapping source1_entity_id -> set of candidate entity IDs."""
    df = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False)
    cand_map: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cands_str = str(row["candidate_entity_ids"]).strip()
        if cands_str:
            cands = {c.strip() for c in cands_str.split(",") if c.strip()}
            cand_map[s1_id] = cands
        else:
            cand_map[s1_id] = set()
    return cand_map


def diagnose_missed_matches(
    candidates_path: str,
    ground_truth_path: str,
    data_dir: Optional[str] = None,
    s1_path: Optional[str] = None,
    s2_path: Optional[str] = None,
    s3_path: Optional[str] = None,
    out_path: str = "output/missed_pairs_diagnostic.tsv",
    sample_limit: int = 5,
):
    print("=== Stage B: Missed-Match Root Cause Diagnostic ===")
    print(f"Candidates File:    {candidates_path}")
    print(f"Ground Truth File:  {ground_truth_path}")

    gt_map = parse_ground_truth(ground_truth_path)
    cand_map = parse_candidates(candidates_path)

    # Find missed pairs
    missed_pairs: List[Tuple[str, str]] = []
    total_true_pairs = 0
    found_true_pairs = 0

    for s1_id, true_targets in gt_map.items():
        if s1_id not in cand_map:
            continue
        cands = cand_map[s1_id]
        for target_id in true_targets:
            total_true_pairs += 1
            if target_id in cands:
                found_true_pairs += 1
            else:
                missed_pairs.append((s1_id, target_id))

    print(f"\nTotal Ground Truth Pairs in Evaluated Set: {total_true_pairs:,}")
    print(f"Successfully Retained Candidate Pairs:     {found_true_pairs:,}")
    print(f"Missed Candidate Pairs to Diagnose:       {len(missed_pairs):,}")

    if not missed_pairs:
        print("\nAll true matches were successfully captured in candidates! No misses to diagnose.")
        return

    # Load source tables if paths provided
    entity_records: Dict[str, dict] = {}
    if data_dir or (s1_path and s2_path and s3_path):
        print("\nLoading and normalizing entity tables for feature inspection...")
        if data_dir:
            sources = load_all_sources(data_dir)
            s1_df = normalize_source(sources["source1"])
            s2_df = normalize_source(sources["source2"])
            s3_df = normalize_source(sources["source3"])
        else:
            s1_df = normalize_source(pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False))
            s2_df = normalize_source(pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False))
            s3_df = normalize_source(pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False))

        name_col = "business_name_clean" if "business_name_clean" in s1_df.columns else "business_name"
        addr_col = "address_clean" if "address_clean" in s1_df.columns else "address"
        country_col = "country_iso" if "country_iso" in s1_df.columns else "country"

        for df, id_col in [(s1_df, "source1_entity_id"), (s2_df, "source2_entity_id"), (s3_df, "source3_entity_id")]:
            if id_col in df.columns:
                for _, row in df.iterrows():
                    eid = str(row[id_col]).strip()
                    entity_records[eid] = {
                        "name": str(row.get(name_col, "")).strip(),
                        "address": str(row.get(addr_col, "")).strip(),
                        "country": str(row.get(country_col, "")).strip(),
                        "city": str(row.get("city", "")).strip(),
                        "raw_row": row.to_dict(),
                    }

    if not entity_records:
        print("\nNote: Source data files not provided or empty.")
        print("To inspect entity names and addresses, provide --data-dir or --s1/--s2/--s3.")
        return

    # Analyze failure causes
    reasons = []
    diagnostic_rows = []

    for s1_id, target_id in missed_pairs:
        s1_rec = entity_records.get(s1_id)
        target_rec = entity_records.get(target_id)

        if not s1_rec or not target_rec:
            cause = "RECORD_NOT_FOUND_IN_DATA"
            reasons.append(cause)
            diagnostic_rows.append({
                "s1_id": s1_id,
                "target_id": target_id,
                "primary_cause": cause,
                "s1_name": s1_rec["name"] if s1_rec else "",
                "target_name": target_rec["name"] if target_rec else "",
                "s1_country": s1_rec["country"] if s1_rec else "",
                "target_country": target_rec["country"] if target_rec else "",
                "shared_name_tokens": "",
                "char_ngram_jaccard": 0.0,
            })
            continue

        s1_name = s1_rec["name"]
        target_name = target_rec["name"]
        s1_country = s1_rec["country"].upper()
        target_country = target_rec["country"].upper()

        s1_tokens = get_word_tokens(s1_name)
        target_tokens = get_word_tokens(target_name)
        shared_tokens = s1_tokens.intersection(target_tokens)

        s1_ngrams = get_char_ngrams(s1_name, n=3)
        target_ngrams = get_char_ngrams(target_name, n=3)
        ngram_union = s1_ngrams.union(target_ngrams)
        char_jaccard = (len(s1_ngrams.intersection(target_ngrams)) / len(ngram_union)) if ngram_union else 0.0

        s1_addr_tokens = get_word_tokens(s1_rec["address"])
        target_addr_tokens = get_word_tokens(target_rec["address"])
        shared_addr_tokens = s1_addr_tokens.intersection(target_addr_tokens)

        # Categorize primary failure cause
        if not s1_country or not target_country or s1_country == "UNKNOWN" or target_country == "UNKNOWN":
            cause = "MISSING_OR_UNKNOWN_COUNTRY"
        elif s1_country != target_country:
            cause = "COUNTRY_PARTITION_MISMATCH"
        elif s1_name.lower() == target_name.lower():
            cause = "EXACT_NAME_CAPPED_OR_UNSCORED"
        elif len(shared_tokens) > 0:
            # Word tokens match, but was blocked out (high DF, token cap >100, or trimmed score)
            cause = "SHARED_TOKEN_CAPPED_OUT"
        elif char_jaccard >= 0.35:
            # High subword similarity, but zero shared complete tokens (typo / inflection / spelling variation)
            cause = "ZERO_TOKEN_OVERLAP_HIGH_CHAR_SIM"
        elif len(shared_addr_tokens) >= 2:
            # Address shares tokens, but name had no overlap
            cause = "ADDRESS_OVERLAP_ONLY"
        else:
            cause = "SEVERE_NAME_DIVERGENCE"

        reasons.append(cause)
        diagnostic_rows.append({
            "s1_id": s1_id,
            "target_id": target_id,
            "primary_cause": cause,
            "s1_name": s1_name,
            "target_name": target_name,
            "s1_country": s1_country,
            "target_country": target_country,
            "shared_name_tokens": ",".join(sorted(shared_tokens)),
            "char_ngram_jaccard": round(char_jaccard, 3),
            "shared_addr_tokens": ",".join(sorted(shared_addr_tokens)),
        })

    # Summary Report
    counter = Counter(reasons)
    total_missed = len(reasons)

    print("\n" + "=" * 70)
    print("MISSED-MATCH ROOT CAUSE TAXONOMY:")
    print("=" * 70)
    for cause, count in counter.most_common():
        pct = count / total_missed * 100.0
        print(f"  {cause:<35} : {count:>5} ({pct:>5.1f}%)")
    print("=" * 70)

    # Save to TSV
    diag_df = pd.DataFrame(diagnostic_rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    diag_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nDetailed diagnostics saved to: {out_path}")

    # Print sample cases per category
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES PER ROOT CAUSE CATEGORY:")
    print("=" * 70)
    for cause, _ in counter.most_common():
        subset = [r for r in diagnostic_rows if r["primary_cause"] == cause]
        print(f"\n--- Category: {cause} (showing up to {sample_limit} examples) ---")
        for sample in subset[:sample_limit]:
            print(f"  S1 ID:     {sample['s1_id']} | Country: {sample['s1_country']} | Name: {sample['s1_name']}")
            print(f"  Target ID: {sample['target_id']} | Country: {sample['target_country']} | Name: {sample['target_name']}")
            print(f"  Shared Tokens: '{sample['shared_name_tokens']}' | Char Jaccard: {sample['char_ngram_jaccard']}")
            if sample.get("shared_addr_tokens"):
                print(f"  Shared Address Tokens: '{sample['shared_addr_tokens']}'")
            print()


def main():
    parser = argparse.ArgumentParser(description="Stage B Missed-Match Diagnostic")
    parser.add_argument(
        "--candidates",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Path to candidate_pairs.tsv",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="dataset/train/train_ground_truth.tsv",
        help="Path to ground truth TSV",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing source1.tsv, source2.tsv, source3.tsv",
    )
    parser.add_argument(
        "--s1",
        type=str,
        default=None,
        help="Path to source1 TSV",
    )
    parser.add_argument(
        "--s2",
        type=str,
        default=None,
        help="Path to source2 TSV",
    )
    parser.add_argument(
        "--s3",
        type=str,
        default=None,
        help="Path to source3 TSV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/missed_pairs_diagnostic.tsv",
        help="Output TSV path for diagnostic breakdown",
    )
    args = parser.parse_args()

    diagnose_missed_matches(
        candidates_path=args.candidates,
        ground_truth_path=args.ground_truth,
        data_dir=args.data_dir,
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

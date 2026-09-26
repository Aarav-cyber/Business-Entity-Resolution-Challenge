"""
src/blocking/evaluate_blocking.py
Author: Ankush (Stage B - Blocking & Candidate Generation)

Diagnostic and evaluation utility to compute Stage B Recall Ceiling and candidate metrics
against a ground-truth mapping file.

Usage:
  python src/blocking/evaluate_blocking.py \
      --candidates output/candidate_pairs.tsv \
      --ground-truth dataset/train/train_ground_truth.tsv
"""

import argparse
import sys
from typing import Dict, List, Set
import numpy as np
import pandas as pd


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


def evaluate_blocking(
    candidates_path: str,
    ground_truth_path: str,
    max_miss_samples: int = 10,
):
    print("=== Stage B: Evaluating Blocking Recall Ceiling & Quality ===")
    print(f"Candidates File:    {candidates_path}")
    print(f"Ground Truth File:  {ground_truth_path}")

    gt_map = parse_ground_truth(ground_truth_path)
    cand_map = parse_candidates(candidates_path)

    # Filter to S1 entities present in both
    eval_s1_ids = [s1 for s1 in gt_map if s1 in cand_map]
    if not eval_s1_ids:
        print("Error: No overlapping Source 1 IDs found between candidates and ground truth!")
        sys.exit(1)

    total_true_matches = 0
    found_true_matches = 0
    full_coverage_s1 = 0
    partial_coverage_s1 = 0
    total_s1_with_matches = 0

    candidate_counts = []
    missed_pairs: List[Tuple[str, str]] = []

    for s1_id in eval_s1_ids:
        true_set = gt_map[s1_id]
        cand_set = cand_map[s1_id]

        candidate_counts.append(len(cand_set))

        if not true_set:
            # Singleton entity (no matches expected)
            continue

        total_s1_with_matches += 1
        total_true_matches += len(true_set)

        intersection = true_set.intersection(cand_set)
        found_true_matches += len(intersection)

        if len(intersection) == len(true_set):
            full_coverage_s1 += 1
        if len(intersection) > 0:
            partial_coverage_s1 += 1

        diff = true_set - cand_set
        for missed_cand in diff:
            if len(missed_pairs) < max_miss_samples:
                missed_pairs.append((s1_id, missed_cand))

    recall_ceiling = (found_true_matches / total_true_matches * 100.0) if total_true_matches > 0 else 100.0
    full_cov_pct = (full_coverage_s1 / total_s1_with_matches * 100.0) if total_s1_with_matches > 0 else 100.0
    partial_cov_pct = (partial_coverage_s1 / total_s1_with_matches * 100.0) if total_s1_with_matches > 0 else 100.0

    counts = np.array(candidate_counts)

    print("\n" + "=" * 60)
    print("STAGE B RECALL & COVERAGE REPORT:")
    print("=" * 60)
    print(f"Evaluated S1 Entities:     {len(eval_s1_ids):,}")
    print(f"S1 Entities with Matches:  {total_s1_with_matches:,}")
    print(f"Total True Matches:        {total_true_matches:,}")
    print(f"Found True Matches:        {found_true_matches:,}")
    print(f"RECALL CEILING:            {recall_ceiling:.3f}%")
    print(f"Full Entity Coverage:      {full_cov_pct:.2f}% (100% matches found)")
    print(f"Partial Entity Coverage:   {partial_cov_pct:.2f}% (>=1 match found)")
    print("-" * 60)
    print("CANDIDATE DISTRIBUTION:")
    print(f"Average Candidate Count:   {np.mean(counts):.2f}")
    print(f"Median Candidate Count:    {np.median(counts):.1f}")
    print(f"P90 Candidate Count:       {np.percentile(counts, 90):.1f}")
    print(f"P95 Candidate Count:       {np.percentile(counts, 95):.1f}")
    print(f"P99 Candidate Count:       {np.percentile(counts, 99):.1f}")
    print(f"Max Candidate Count:       {np.max(counts)}")
    print("=" * 60 + "\n")

    if missed_pairs:
        print("SAMPLE MISSED MATCHES (For Diagnosing Blocker Gaps):")
        for s1_id, missed_cid in missed_pairs:
            print(f"  - Missed Pair: S1={s1_id} <---> Target={missed_cid}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage B Blocking Recall Ceiling")
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
        help="Path to train_ground_truth.tsv",
    )
    args = parser.parse_args()

    evaluate_blocking(args.candidates, args.ground_truth)


if __name__ == "__main__":
    main()

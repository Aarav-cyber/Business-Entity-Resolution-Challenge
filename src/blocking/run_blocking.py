"""
src/blocking/run_blocking.py
Author: Ankush (Stage B - Blocking & Candidate Generation)

CLI script to run Stage B blocking on entity resolution datasets and output candidate_pairs.tsv.

Usage:
  python src/blocking/run_blocking.py --data-dir dataset/test --out output/candidate_pairs.tsv
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# Add repo root to import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_all_sources
from src.data.normalize import normalize_source
from src.blocking.blocking import BlockingEngine, write_candidate_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Stage B Candidate Generation / Blocking")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/test",
        help="Directory containing source1, source2, and source3 TSV files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output/candidate_pairs.tsv",
        help="Output path for candidate_pairs.tsv",
    )
    parser.add_argument(
        "--max-token-df",
        type=int,
        default=15000,
        help="Max document frequency for rare token indexing",
    )
    parser.add_argument(
        "--min-token-idf",
        type=float,
        default=2.0,
        help="Minimum smoothed IDF for rare token indexing",
    )
    parser.add_argument(
        "--top-k-tfidf",
        type=int,
        default=25,
        help="Top-K nearest neighbors retrieved via TF-IDF cosine per partition",
    )
    parser.add_argument(
        "--tfidf-max-dist",
        type=float,
        default=0.65,
        help="Max cosine distance threshold for TF-IDF neighbor acceptance",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional max candidates per S1 entity (None = unlimited / no truncation)",
    )
    args = parser.parse_args()

    logger.info("=== Stage B: Running Blocking / Candidate Generation ===")
    logger.info(f"Data Directory: {args.data_dir}")
    logger.info(f"Output File:    {args.out}")

    start_time = time.time()

    # 1. Load Data
    sources = load_all_sources(args.data_dir)
    s1_raw = sources["source1"]
    s2_raw = sources["source2"]
    s3_raw = sources["source3"]

    logger.info(f"Loaded records: S1={len(s1_raw):,}, S2={len(s2_raw):,}, S3={len(s3_raw):,}")

    # 2. Normalize Data using Stage A Engine
    logger.info("Applying Stage A Normalization...")
    s1_clean = normalize_source(s1_raw)
    s2_clean = normalize_source(s2_raw)
    s3_clean = normalize_source(s3_raw)

    # 3. Initialize & Fit Blocking Engine
    engine = BlockingEngine(
        max_token_df=args.max_token_df,
        min_token_idf=args.min_token_idf,
        top_k_tfidf=args.top_k_tfidf,
        tfidf_max_dist=args.tfidf_max_dist,
        max_candidates_per_s1=args.max_candidates,
    )

    logger.info("Fitting global token statistics...")
    engine.fit_token_stats([s1_clean, s2_clean, s3_clean])

    # 4. Generate Candidate Pairs
    logger.info("Generating candidates across country partitions with relevance ranking...")
    candidates_dict, provenance = engine.generate_candidates(s1_clean, s2_clean, s3_clean)

    # 5. Compute Candidate Set Metrics
    total_s1 = len(candidates_dict)
    counts = [len(cands) for cands in candidates_dict.values()]
    total_candidates = sum(counts)
    avg_candidates = total_candidates / max(total_s1, 1)
    median_candidates = float(np.median(counts)) if counts else 0.0
    p90_candidates = float(np.percentile(counts, 90)) if counts else 0.0
    p95_candidates = float(np.percentile(counts, 95)) if counts else 0.0
    p99_candidates = float(np.percentile(counts, 99)) if counts else 0.0
    max_candidates = max(counts) if counts else 0

    total_possible_pairs = len(s1_clean) * (len(s2_clean) + len(s3_clean))
    reduction_ratio = (
        1.0 - (total_candidates / total_possible_pairs)
        if total_possible_pairs > 0
        else 1.0
    )

    print("\n" + "=" * 55)
    print("STAGE B BLOCKING METRICS SUMMARY:")
    print("=" * 55)
    print(f"Total Source 1 Entities:   {total_s1:,}")
    print(f"Total Candidate Pairs:     {total_candidates:,}")
    print(f"Average Candidate Size:    {avg_candidates:.2f} per S1 entity")
    print(f"Median Candidate Size:     {median_candidates:.1f}")
    print(f"P90 Candidate Size:        {p90_candidates:.1f}")
    print(f"P95 Candidate Size:        {p95_candidates:.1f}")
    print(f"P99 Candidate Size:        {p99_candidates:.1f}")
    print(f"Max Candidate Size:        {max_candidates}")
    print(f"Reduction Ratio:           {reduction_ratio * 100:.6f}%")
    print(f"Total Execution Time:      {time.time() - start_time:.2f}s")
    print("=" * 55 + "\n")

    # 6. Write candidate_pairs.tsv Output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_candidate_pairs(candidates_dict, args.out)
    logger.info(f"Successfully wrote candidate pairs to: {args.out}")


if __name__ == "__main__":
    main()

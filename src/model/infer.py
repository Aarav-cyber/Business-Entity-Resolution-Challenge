"""
src/model/infer.py
Author: Riju (Stage C - Feature Engineering & Matching Model)

Scores every candidate in output/candidate_pairs.tsv (Stage B) with
features.pair_score against the calibrated threshold (train.py) and
writes output/matching_results.tsv, per the CONTRACTS.md schema:
one row per Source 1 entity, matched_entity_ids a strict subset of
candidate_entity_ids.

Usage:
  python src/model/infer.py --data-dir dataset/test \
      --candidates output/candidate_pairs.tsv \
      --threshold src/model/threshold.json \
      --out output/matching_results.tsv
"""
import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_all_sources
from src.data.normalize import normalize_source
from src.model.features import record_signature, pair_score


def main():
    ap = argparse.ArgumentParser(description="Run Stage C inference")
    ap.add_argument("--data-dir", default="dataset/test")
    ap.add_argument("--candidates", default="output/candidate_pairs.tsv")
    ap.add_argument("--threshold", default="src/model/threshold.json")
    ap.add_argument("--out", default="output/matching_results.tsv")
    args = ap.parse_args()

    thr = 0.55
    if os.path.exists(args.threshold):
        with open(args.threshold) as f:
            thr = json.load(f)["threshold"]

    sources = load_all_sources(args.data_dir)
    s1 = normalize_source(sources["source1"])
    s2 = normalize_source(sources["source2"])
    s3 = normalize_source(sources["source3"])

    sig1 = {r.entity_id: record_signature(r.business_name_clean, r.business_address_clean) for r in s1.itertuples()}
    sig23 = {r.entity_id: record_signature(r.business_name_clean, r.business_address_clean) for r in s2.itertuples()}
    sig23.update({r.entity_id: record_signature(r.business_name_clean, r.business_address_clean) for r in s3.itertuples()})

    rows = []
    seen = set()
    with open(args.candidates, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            s1id = row[0].strip()
            cand_str = row[1].strip() if len(row) > 1 else ""
            seen.add(s1id)
            matched = []
            if s1id in sig1 and cand_str:
                s1sig = sig1[s1id]
                scored = [(pair_score(s1sig, sig23[c]), c) for c in cand_str.split(",") if c and c in sig23]
                matched = [c for s, c in sorted(scored, reverse=True) if s >= thr]
            rows.append((s1id, matched))

    # every Source 1 test entity must appear, even if absent from candidates
    for eid in s1.entity_id:
        if eid not in seen:
            rows.append((eid, []))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1id, matched in rows:
            w.writerow([s1id, ",".join(matched)])

    n_matched = sum(1 for _, m in rows if m)
    print(f"threshold={thr:.2f} rows={len(rows)} with_match={n_matched} -> {args.out}")


if __name__ == "__main__":
    main()

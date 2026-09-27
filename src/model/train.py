"""
src/model/train.py
Author: Riju (Stage C - Feature Engineering & Matching Model)

Calibrates a single decision threshold on `pair_score` (features.py)
using train_ground_truth.tsv. No classifier is fit - with ~250-300
ground-truth pairs actually surviving the train S1/S2/S3 sampling
mismatch (documented in Documentation_template.md), a threshold search
over one composite score generalizes better than a learned model would
on so few positives.

Usage:
  python src/model/train.py --data-dir dataset/train --out src/model/threshold.json
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.loader import load_all_sources, load_ground_truth
from src.data.normalize import normalize_source
from src.model.features import record_signature, pair_score

random.seed(42)


def f_beta(p, r, beta=0.5):
    if p == 0 and r == 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * p + r
    return 0.0 if denom == 0 else (1 + b2) * p * r / denom


def build_signatures(df):
    return {
        row.entity_id: record_signature(row.business_name_clean, row.business_address_clean)
        for row in df.itertuples()
    }, dict(zip(df.entity_id, df.country))


def main():
    ap = argparse.ArgumentParser(description="Calibrate Stage C match threshold")
    ap.add_argument("--data-dir", default="dataset/train")
    ap.add_argument("--out", default="src/model/threshold.json")
    ap.add_argument("--neg-per-pos", type=int, default=8)
    args = ap.parse_args()

    sources = load_all_sources(args.data_dir)
    s1 = normalize_source(sources["source1"])
    s2 = normalize_source(sources["source2"])
    s3 = normalize_source(sources["source3"])
    gt = load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))

    sig1, _ = build_signatures(s1)
    sig2, country2 = build_signatures(s2)
    sig3, country3 = build_signatures(s3)

    def lookup(eid):
        if eid.startswith("S2-"):
            return sig2.get(eid), country2.get(eid)
        if eid.startswith("S3-"):
            return sig3.get(eid), country3.get(eid)
        return None, None

    by_country = defaultdict(list)
    for eid, c in {**country2, **country3}.items():
        by_country[c].append(eid)

    pos_scores, neg_scores = [], []
    for row in gt.itertuples():
        if row.source1_entity_id not in sig1 or not row.matched_entity_ids:
            continue
        matched = [m for m in row.matched_entity_ids.split(",") if m]
        s1sig = sig1[row.source1_entity_id]
        for mid in matched:
            msig, _ = lookup(mid)
            if msig is not None:
                pos_scores.append(pair_score(s1sig, msig))
        # sample same-country negatives from outside the true match set
        country = s1.loc[s1.entity_id == row.source1_entity_id, "country"].iloc[0]
        pool = by_country.get(country, [])
        got, tries = 0, 0
        while pool and got < args.neg_per_pos and tries < args.neg_per_pos * 4:
            tries += 1
            cand = random.choice(pool)
            if cand in matched:
                continue
            csig, _ = lookup(cand)
            if csig is not None:
                neg_scores.append(pair_score(s1sig, csig))
                got += 1

    best_thr, best_f = 0.5, -1.0
    for i in range(101):
        thr = i / 100.0
        tp = sum(s >= thr for s in pos_scores)
        fp = sum(s >= thr for s in neg_scores)
        fn = len(pos_scores) - tp
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f = f_beta(prec, rec)
        if f > best_f:
            best_f, best_thr = f, thr

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "threshold": best_thr,
            "calibration_f0.5": round(best_f, 4),
            "n_positive_pairs": len(pos_scores),
            "n_negative_pairs": len(neg_scores),
        }, f, indent=2)

    print(f"n_pos={len(pos_scores)} n_neg={len(neg_scores)} "
          f"threshold={best_thr:.2f} calibration_F0.5={best_f:.3f} -> {args.out}")


if __name__ == "__main__":
    main()

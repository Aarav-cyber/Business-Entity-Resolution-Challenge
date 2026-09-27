"""
stage_c_full_scale.py
Run in Colab against your FULL train/test files (millions of rows).
Streams everything - never loads a whole 5M-row source file into memory
at once - so it's safe on standard Colab RAM.

USAGE (run both, in order):

  python3 stage_c_full_scale.py calibrate \
      --train-dir /content/dataset/train \
      --out /content/threshold.json

  python3 stage_c_full_scale.py infer \
      --test-dir /content/dataset/test \
      --candidates /content/output/candidate_pairs.tsv \
      --threshold /content/threshold.json \
      --out /content/output/matching_results.tsv

--train-dir must contain: train_source1.tsv, train_source2.tsv,
    train_source3.tsv, train_ground_truth.tsv
--test-dir must contain: test_source1.tsv, test_source2.tsv, test_source3.tsv
--candidates is YOUR FULL candidate_pairs.tsv (all ~916K S1 rows) - the
    100K sample can't produce a complete matching_results.tsv, which is
    exactly the "missing entities" error you hit.
"""
import argparse, csv, json, os, random, re, sys
from collections import defaultdict

random.seed(42)
csv.field_size_limit(sys.maxsize)

STOP_SUFFIX = set("ltd limited pvt private inc incorporated corp corporation "
                   "llc llp co company plc gmbh sa sarl srl bv nv pty".split())
ADDR_STOP = set("road rd street st near post po box village town city district "
                "state country floor flat no number unit apartment apartments "
                "colony nagar main cross".split())


def norm(s):
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def bigrams(s):
    s = s.replace(" ", "")
    return set(s[i:i+2] for i in range(len(s)-1)) if len(s) > 1 else ({s} if s else set())


def signature(name, addr):
    n = norm(name)
    a = norm(addr)
    return {
        "nb": bigrams(n),
        "nt": {t for t in n.split() if t not in STOP_SUFFIX and len(t) > 1},
        "ab": bigrams(" ".join(t for t in a.split() if t not in ADDR_STOP)),
    }


def dice(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def score(s1, s2):
    return 0.55 * dice(s1["nb"], s2["nb"]) + 0.25 * jaccard(s1["nt"], s2["nt"]) + 0.20 * dice(s1["ab"], s2["ab"])


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        idx = {c.strip(): i for i, c in enumerate(header)}
        for row in r:
            if len(row) < 4:
                continue
            yield row, idx


def f_beta(p, r, beta=0.5):
    if p == 0 and r == 0: return 0.0
    b2 = beta * beta
    d = b2 * p + r
    return 0.0 if d == 0 else (1 + b2) * p * r / d


def cmd_calibrate(args):
    # pass 1: read ground truth, note which S1 have matches + their true match ids
    s1_matches = {}
    with open(os.path.join(args.train_dir, "train_ground_truth.tsv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if not row: continue
            s1id = row[0].strip()
            ids = [x for x in (row[1].strip() if len(row) > 1 else "").split(",") if x]
            if ids:
                s1_matches[s1id] = ids

    needed_true = set(i for ids in s1_matches.values() for i in ids)
    print(f"ground-truth S1 with matches: {len(s1_matches)}, referenced S2/S3 ids: {len(needed_true)}")

    # pass 2: reservoir-sample a negative pool per country from S2+S3, and
    # grab records for needed_true + S1 entities that have matches
    pool_size_per_country = 200_000
    neg_pool = defaultdict(list)  # country -> [(eid, name, addr)]
    needed_records = {}  # eid -> (name, addr)

    for fname in ("train_source2.tsv", "train_source3.tsv"):
        path = os.path.join(args.train_dir, fname)
        if not os.path.exists(path):
            continue
        for row, idx in read_tsv(path):
            eid, name, addr, country = row[idx["entity_id"]], row[idx["business_name"]], row[idx["business_address"]], row[idx["country"]].strip()
            if eid in needed_true:
                needed_records[eid] = (name, addr)
            pool = neg_pool[country]
            if len(pool) < pool_size_per_country:
                pool.append((eid, name, addr))
            else:
                j = random.randint(0, pool_size_per_country * 3)  # cheap approx reservoir
                if j < pool_size_per_country:
                    pool[j] = (eid, name, addr)

    s1_records = {}
    for row, idx in read_tsv(os.path.join(args.train_dir, "train_source1.tsv")):
        eid = row[idx["entity_id"]]
        if eid in s1_matches:
            s1_records[eid] = (row[idx["business_name"]], row[idx["business_address"]], row[idx["country"]].strip())

    pos_scores, neg_scores = [], []
    for s1id, matched in s1_matches.items():
        if s1id not in s1_records:
            continue
        name, addr, country = s1_records[s1id]
        sig1 = signature(name, addr)
        surviving = [m for m in matched if m in needed_records]
        for mid in surviving:
            mname, maddr = needed_records[mid]
            pos_scores.append(score(sig1, signature(mname, maddr)))
        pool = neg_pool.get(country, [])
        if not pool:
            continue
        got, tries = 0, 0
        while got < args.neg_per_pos and tries < args.neg_per_pos * 4 and pool:
            tries += 1
            eid, nname, naddr = random.choice(pool)
            if eid in matched:
                continue
            neg_scores.append(score(sig1, signature(nname, naddr)))
            got += 1

    print(f"usable positive pairs: {len(pos_scores)}  negative samples: {len(neg_scores)}")
    if not pos_scores:
        print("WARNING: zero usable positive pairs survived - falling back to default threshold 0.55")
        thr, f = 0.55, 0.0
    else:
        best_thr, best_f = 0.5, -1.0
        for i in range(101):
            t = i / 100.0
            tp = sum(s >= t for s in pos_scores)
            fp = sum(s >= t for s in neg_scores)
            fn = len(pos_scores) - tp
            p = tp / (tp + fp) if (tp + fp) else 1.0
            rc = tp / (tp + fn) if (tp + fn) else 0.0
            fscore = f_beta(p, rc)
            if fscore > best_f:
                best_f, best_thr = fscore, t
        thr, f = best_thr, best_f

    with open(args.out, "w") as fh:
        json.dump({"threshold": thr, "calibration_f0.5": round(f, 4),
                    "n_positive_pairs": len(pos_scores), "n_negative_pairs": len(neg_scores)}, fh, indent=2)
    print(f"threshold={thr:.2f} calibration_F0.5={f:.3f} -> {args.out}")


def _build_sqlite_index(conn, table, paths):
    """Stream TSV file(s) into a disk-backed SQLite table instead of a
    Python dict, so a 5M-row source never has to fit in RAM at once."""
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TABLE {table} (entity_id TEXT PRIMARY KEY, name TEXT, addr TEXT)")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA journal_mode=MEMORY")
    batch, BATCH = [], 20000
    n = 0
    for path in paths:
        for row, idx in read_tsv(path):
            batch.append((row[idx["entity_id"]], row[idx["business_name"]], row[idx["business_address"]]))
            if len(batch) >= BATCH:
                conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)", batch)
                n += len(batch)
                batch = []
    if batch:
        conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)", batch)
        n += len(batch)
    conn.commit()
    print(f"  indexed {n} rows into table '{table}'")


def cmd_infer(args):
    thr = 0.55
    if os.path.exists(args.threshold):
        thr = json.load(open(args.threshold))["threshold"]
    print(f"using threshold={thr}")

    import sqlite3
    if os.path.exists(args.db):
        os.remove(args.db)
    conn = sqlite3.connect(args.db)
    print("Building on-disk index (bounded RAM, uses disk at --db)...")
    _build_sqlite_index(conn, "s1", [os.path.join(args.test_dir, "test_source1.tsv")])
    _build_sqlite_index(conn, "cand", [os.path.join(args.test_dir, "test_source2.tsv"),
                                        os.path.join(args.test_dir, "test_source3.tsv")])
    cur = conn.cursor()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    written = set()  # only entity_id strings, not full records - small
    n_matched = 0
    n_rows = 0
    with open(args.candidates, newline="", encoding="utf-8") as fin, \
         open(args.out, "w", newline="", encoding="utf-8") as fout:
        r = csv.reader(fin, delimiter="\t")
        next(r)
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        for row in r:
            if not row: continue
            s1id = row[0].strip()
            cand_str = row[1].strip() if len(row) > 1 else ""
            matched = []
            if cand_str:
                cur.execute("SELECT name, addr FROM s1 WHERE entity_id=?", (s1id,))
                s1rec = cur.fetchone()
                cand_ids = [c for c in cand_str.split(",") if c]
                if s1rec and cand_ids:
                    sig1 = signature(s1rec[0], s1rec[1])
                    qmarks = ",".join("?" * len(cand_ids))
                    cur.execute(f"SELECT entity_id, name, addr FROM cand WHERE entity_id IN ({qmarks})", cand_ids)
                    scored = [(score(sig1, signature(name, addr)), eid) for eid, name, addr in cur.fetchall()]
                    matched = [c for s, c in sorted(scored, reverse=True) if s >= thr]
            if matched:
                n_matched += 1
            w.writerow([s1id, ",".join(matched)])
            written.add(s1id)
            n_rows += 1
            if n_rows % 100000 == 0:
                print(f"  scored {n_rows} S1 rows...")

        # THIS is what fixes your "missing entities" error: stream test
        # S1 again (not from RAM - a fresh cursor) and emit an empty row
        # for any id that never appeared in candidate_pairs.tsv at all.
        missing = 0
        cur2 = conn.cursor()
        cur2.execute("SELECT entity_id FROM s1")
        for (eid,) in cur2.fetchall():
            if eid not in written:
                w.writerow([eid, ""])
                missing += 1

    total = n_rows + missing
    conn.close()
    print(f"rows written: {total} (of which {missing} had no candidate-file row -> empty)")
    print(f"rows with >=1 match: {n_matched}")
    print(f"-> {args.out}")
    print(f"(you can delete the index db now: {args.db})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--train-dir", required=True)
    c.add_argument("--out", default="threshold.json")
    c.add_argument("--neg-per-pos", type=int, default=8)
    c.set_defaults(func=cmd_calibrate)

    i = sub.add_parser("infer")
    i.add_argument("--test-dir", required=True)
    i.add_argument("--candidates", required=True)
    i.add_argument("--threshold", default="threshold.json")
    i.add_argument("--out", default="output/matching_results.tsv")
    i.add_argument("--db", default="/content/stage_c_index.sqlite",
                    help="scratch SQLite file used to index records on disk instead of RAM; safe to delete after")
    i.set_defaults(func=cmd_infer)

    args = ap.parse_args()
    args.func(args)

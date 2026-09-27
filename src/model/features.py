"""
src/model/features.py
Author: Riju (Stage C - Feature Engineering & Matching Model)

Pairwise similarity features between a Source 1 record and a candidate
S2/S3 record. Operates on Stage A's `business_name_clean` /
`business_address_clean` columns (see src/data/normalize.py); no
re-normalization happens here.

Pure-Python set ops (no sklearn/scipy) keep this fast and dependency
light for scoring millions of candidate pairs on a laptop.
"""

STOP_SUFFIX = set("ltd limited pvt private inc incorporated corp corporation "
                   "llc llp co company plc gmbh sa sarl srl bv nv pty".split())
ADDR_STOP = set("road rd street st near post po box village town city district "
                 "state country floor flat no number unit apartment apartments "
                 "colony nagar main cross".split())


def _bigrams(text: str) -> set:
    s = text.replace(" ", "")
    return set(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else ({s} if s else set())


def name_tokens(name_clean: str) -> set:
    return {t for t in name_clean.split() if t not in STOP_SUFFIX and len(t) > 1}


def addr_bigrams(addr_clean: str) -> set:
    filtered = " ".join(t for t in addr_clean.split() if t not in ADDR_STOP)
    return _bigrams(filtered)


def dice(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def record_signature(name_clean: str, addr_clean: str) -> dict:
    """Precompute per-record feature ingredients once; reused across
    every candidate pair that record takes part in."""
    return {
        "name_bg": _bigrams(name_clean),
        "name_tok": name_tokens(name_clean),
        "addr_bg": addr_bigrams(addr_clean),
    }


def pair_score(sig1: dict, sig2: dict) -> float:
    """Combined match score in [0, 1]. Name-dominant, address secondary:
    address boilerplate (shared city/state/country terms) is stripped in
    addr_bigrams to avoid false-positive inflation on countries with no
    labeled training data, e.g. France."""
    name_dice = dice(sig1["name_bg"], sig2["name_bg"])
    name_jac = jaccard(sig1["name_tok"], sig2["name_tok"])
    addr_dice = dice(sig1["addr_bg"], sig2["addr_bg"])
    return 0.55 * name_dice + 0.25 * name_jac + 0.20 * addr_dice

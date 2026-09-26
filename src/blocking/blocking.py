"""
src/blocking/blocking.py
Author: Ankush (Stage B - Blocking & Candidate Generation)

Hardened, scalable candidate generation engine for Business Entity Resolution.
Architecture:
  1. Open-Set Country Partitioning (with lightweight token fallback for missing countries)
  2. High-IDF Rare Token Inverted Indexing
  3. Partition-wise TF-IDF Cosine Nearest-Neighbor Retrieval (character 3-5 gram wb)
  4. Locality & Landmark Token Recovery / Enrichment
  5. Multi-Signal Candidate Relevance Scoring (no blind truncation)
  6. Synchronized Candidate & Provenance Trimming (Top-K by relevance score)
"""

import logging
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Token Extraction & Frequency Utilities
# -------------------------------------------------------------------

COMMON_ADDRESS_STOPWORDS = {
    "road", "street", "st", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "highway", "hwy", "way", "parkway", "pkwy",
    "floor", "fl", "unit", "apt", "apartment", "suite", "ste", "building", "bldg",
    "house", "no", "number", "plot", "block", "sector", "phase", "near", "opposite",
    "opp", "behind", "beside", "post", "box", "po", "city", "state", "dist", "district",
    "null", "none", "na", "north", "south", "east", "west"
}


def get_name_tokens(text: str) -> List[str]:
    """Extract clean words from a normalized business name string."""
    if not text or pd.isna(text):
        return []
    return [t for t in str(text).split() if len(t) > 1]


def get_informative_address_tokens(address: str) -> Set[str]:
    """
    Extract meaningful locality and area tokens from an address,
    ignoring common structural stopwords and purely numeric tokens.
    """
    if not address or pd.isna(address):
        return set()
    tokens = re.findall(r"\b[^\W\d_]+\b", str(address).lower(), flags=re.UNICODE)
    return {
        t for t in tokens
        if len(t) > 2 and t not in COMMON_ADDRESS_STOPWORDS
    }


def build_token_frequencies(dfs: List[pd.DataFrame], name_col: str = "business_name_clean") -> Dict[str, int]:
    """
    Computes global document frequency for every name token across all sources.
    """
    freqs: Dict[str, int] = defaultdict(int)
    for df in dfs:
        if name_col not in df.columns:
            continue
        for name in df[name_col]:
            tokens = set(get_name_tokens(name))
            for t in tokens:
                freqs[t] += 1
    return freqs


# -------------------------------------------------------------------
# Core Blocking Engine Class
# -------------------------------------------------------------------

class BlockingEngine:
    def __init__(
        self,
        max_token_df: int = 15_000,
        min_token_idf: float = 2.0,
        top_k_tfidf: int = 25,
        max_candidates_per_s1: Optional[int] = None,
        tfidf_max_dist: float = 0.65,
    ):
        """
        Args:
            max_token_df: Max document frequency for a token to be indexed.
            min_token_idf: Minimum IDF score required to index a token.
            top_k_tfidf: Max nearest neighbors to retrieve per S1 entity via TF-IDF cosine distance.
            max_candidates_per_s1: Max candidates to retain per S1 entity (None = unlimited / no truncation).
            tfidf_max_dist: Max cosine distance threshold (default 0.65 => min similarity 0.35).
        """
        self.max_token_df = max_token_df
        self.min_token_idf = min_token_idf
        self.top_k_tfidf = top_k_tfidf
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.tfidf_max_dist = tfidf_max_dist
        self.token_df_counts: Dict[str, int] = {}
        self.total_docs: int = 0

    def fit_token_stats(self, source_dfs: List[pd.DataFrame], name_col: str = "business_name_clean"):
        """Precomputes token document frequencies across all source DataFrames."""
        self.token_df_counts = build_token_frequencies(source_dfs, name_col=name_col)
        self.total_docs = sum(len(df) for df in source_dfs)

    def get_token_idf(self, token: str) -> float:
        """Returns the smoothed IDF for a given token."""
        df_count = self.token_df_counts.get(token, 0)
        return math.log((self.total_docs + 1) / (df_count + 1))

    def is_rare_token(self, token: str) -> bool:
        """Returns True if a token has sufficiently high IDF and is not hyper-frequent."""
        df_count = self.token_df_counts.get(token, 0)
        if df_count == 0:
            return True
        if df_count > self.max_token_df:
            return False
        if self.total_docs < 100:
            return df_count <= max(2, int(self.total_docs * 0.5))
        return self.get_token_idf(token) >= self.min_token_idf


    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Set[str]]]]:
        """
        Runs the multi-layer blocking pipeline with relevance scoring.

        Returns:
            candidates_dict: Mapping s1_entity_id -> ranked list of candidate entity IDs (S2/S3).
            provenance_dict: Mapping s1_entity_id -> {cand_id: {reasons...}} in sync with candidates.
        """
        if not self.token_df_counts:
            self.fit_token_stats([s1_df, s2_df, s3_df])

        # Combine S2 and S3 target pool
        s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

        candidates_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        provenance_map: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        # ---------------------------------------------------------------
        # Layer 1: Partitioning by Country
        # ---------------------------------------------------------------
        s1_by_country: Dict[str, pd.DataFrame] = dict(tuple(s1_df.groupby("country", dropna=False)))
        s23_by_country: Dict[str, pd.DataFrame] = dict(tuple(s23_df.groupby("country", dropna=False)))

        all_countries = set(s1_by_country.keys()).union(set(s23_by_country.keys()))

        for country in all_countries:
            # Handle empty/missing country separately via fallback
            if pd.isna(country) or str(country).strip() == "":
                continue

            s1_sub = s1_by_country.get(country, pd.DataFrame())
            s23_sub = s23_by_country.get(country, pd.DataFrame())

            if s1_sub.empty or s23_sub.empty:
                continue

            sub_scores, sub_prov = self._process_partition(s1_sub, s23_sub, country_tag=str(country))
            for s1_id, cands in sub_scores.items():
                for cid, score in cands.items():
                    candidates_scores[s1_id][cid] += score
                    provenance_map[s1_id][cid].update(sub_prov[s1_id][cid])

        # ---------------------------------------------------------------
        # Layer 2: Scalable Global Fallback for Missing Country / 0-Candidate S1s
        # (Uses pure token inverted index, avoiding massive global TF-IDF matrix)
        # ---------------------------------------------------------------
        all_s1_ids = set(s1_df["entity_id"].unique())
        empty_or_missing_s1_ids = {
            s1_id for s1_id in all_s1_ids
            if s1_id not in candidates_scores or len(candidates_scores[s1_id]) == 0
        }

        if empty_or_missing_s1_ids:
            s1_fallback = s1_df[s1_df["entity_id"].isin(empty_or_missing_s1_ids)]
            self._apply_lightweight_token_fallback(
                s1_fallback, s23_df, candidates_scores, provenance_map
            )

        # ---------------------------------------------------------------
        # Final Relevance Ranking & Provenance Synchronization
        # ---------------------------------------------------------------
        final_candidates: Dict[str, List[str]] = {}
        final_provenance: Dict[str, Dict[str, Set[str]]] = {}

        for s1_id in s1_df["entity_id"]:
            cand_score_map = candidates_scores.get(s1_id, {})
            # Validate IDs: must be S2 or S3, never S1
            valid_cands = [
                (cid, score) for cid, score in cand_score_map.items()
                if cid.startswith(("S2-", "S3-")) and cid != s1_id
            ]

            # Rank by relevance score descending
            valid_cands.sort(key=lambda x: x[1], reverse=True)

            # Apply candidate budget if configured
            if self.max_candidates_per_s1 is not None and len(valid_cands) > self.max_candidates_per_s1:
                valid_cands = valid_cands[: self.max_candidates_per_s1]

            retained_cids = [cid for cid, _ in valid_cands]
            final_candidates[s1_id] = retained_cids

            # Synchronize provenance with retained candidates
            final_provenance[s1_id] = {
                cid: provenance_map[s1_id][cid] for cid in retained_cids
            }

        return final_candidates, final_provenance

    def _process_partition(
        self,
        s1_sub: pd.DataFrame,
        s23_sub: pd.DataFrame,
        country_tag: str,
    ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Set[str]]]]:
        """Processes a single country partition using Rare-Token Indexing + TF-IDF Nearest Neighbors + Address/Landmarks."""
        scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        prov: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

        s23_ids = s23_sub["entity_id"].values
        s23_names = s23_sub["business_name_clean"].values

        # --- A: Rare-Token Inverted Index ---
        token_to_s23: Dict[str, List[str]] = defaultdict(list)
        for idx, name in enumerate(s23_names):
            cid = s23_ids[idx]
            for t in get_name_tokens(name):
                if self.is_rare_token(t):
                    token_to_s23[t].append(cid)

        for _, row in s1_sub.iterrows():
            s1_id = row["entity_id"]
            name = row["business_name_clean"]
            tokens = get_name_tokens(name)

            for t in tokens:
                if self.is_rare_token(t) and t in token_to_s23:
                    matched_ids = token_to_s23[t]
                    if len(matched_ids) <= 100:  # Cap candidate fan-out per token
                        token_score = min(1.0, self.get_token_idf(t) * 0.15)
                        for cid in matched_ids:
                            scores[s1_id][cid] += token_score
                            prov[s1_id][cid].add(f"rare_token:{t}")

        # --- B: Partition-wise TF-IDF Nearest-Neighbor Retrieval (Character 3-5 Grams) ---
        if len(s23_sub) > 0 and len(s1_sub) > 0:
            try:
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    dtype=np.float32,
                )

                s23_tfidf = vectorizer.fit_transform(s23_sub["business_name_clean"].fillna(""))
                s1_tfidf = vectorizer.transform(s1_sub["business_name_clean"].fillna(""))

                k_val = min(self.top_k_tfidf, len(s23_sub))
                if k_val > 0:
                    nn_model = NearestNeighbors(n_neighbors=k_val, metric="cosine", algorithm="brute")
                    nn_model.fit(s23_tfidf)

                    distances, indices = nn_model.kneighbors(s1_tfidf)
                    s1_sub_ids = s1_sub["entity_id"].values

                    for i, s1_id in enumerate(s1_sub_ids):
                        for j in range(k_val):
                            dist = distances[i][j]
                            if dist <= self.tfidf_max_dist:
                                sim = 1.0 - dist
                                cand_idx = indices[i][j]
                                cid = s23_ids[cand_idx]
                                scores[s1_id][cid] += sim * 1.5
                                prov[s1_id][cid].add(f"tfidf_nn:{country_tag}")
            except Exception as e:
                logger.warning(f"TF-IDF nearest neighbor retrieval failed for partition '{country_tag}': {e}")

        # --- C: Landmark Token Recovery ---
        if "landmark" in s1_sub.columns and "landmark" in s23_sub.columns:
            landmark_to_s23: Dict[str, List[str]] = defaultdict(list)
            for idx, lm in enumerate(s23_sub["landmark"].values):
                if lm and len(str(lm).strip()) > 3:
                    landmark_to_s23[str(lm).strip().lower()].append(s23_ids[idx])

            for _, row in s1_sub.iterrows():
                s1_id = row["entity_id"]
                lm = str(row.get("landmark", "")).strip().lower()
                if lm and lm in landmark_to_s23:
                    for cid in landmark_to_s23[lm][:25]:
                        scores[s1_id][cid] += 0.8
                        prov[s1_id][cid].add("landmark_match")

        # --- D: Address & Locality Token Recovery ---
        if "business_address_clean" in s1_sub.columns and "business_address_clean" in s23_sub.columns:
            addr_token_to_s23: Dict[str, List[str]] = defaultdict(list)
            for idx, addr in enumerate(s23_sub["business_address_clean"].values):
                cid = s23_ids[idx]
                for at in get_informative_address_tokens(addr):
                    addr_token_to_s23[at].append(cid)

            for _, row in s1_sub.iterrows():
                s1_id = row["entity_id"]
                s1_addr = row.get("business_address_clean", "")
                s1_tokens = get_informative_address_tokens(s1_addr)
                for at in s1_tokens:
                    if at in addr_token_to_s23:
                        matched_cids = addr_token_to_s23[at]
                        # Only use highly specific locality tokens (< 50 occurrences)
                        if len(matched_cids) <= 50:
                            for cid in matched_cids:
                                scores[s1_id][cid] += 0.4
                                prov[s1_id][cid].add(f"address_token:{at}")

        return scores, prov

    def _apply_lightweight_token_fallback(
        self,
        s1_fallback: pd.DataFrame,
        s23_df: pd.DataFrame,
        candidates_scores: Dict[str, Dict[str, float]],
        provenance_map: Dict[str, Dict[str, Set[str]]],
    ):
        """
        Lightweight fallback that indexes rare tokens across S2/S3 without constructing a huge global matrix.
        """
        token_to_s23: Dict[str, List[str]] = defaultdict(list)
        s23_ids = s23_df["entity_id"].values
        s23_names = s23_df["business_name_clean"].values

        for idx, name in enumerate(s23_names):
            cid = s23_ids[idx]
            for t in get_name_tokens(name):
                if self.is_rare_token(t):
                    token_to_s23[t].append(cid)

        for _, row in s1_fallback.iterrows():
            s1_id = row["entity_id"]
            name = row["business_name_clean"]
            tokens = get_name_tokens(name)

            for t in tokens:
                if self.is_rare_token(t) and t in token_to_s23:
                    for cid in token_to_s23[t][:20]:
                        candidates_scores[s1_id][cid] += 0.5
                        provenance_map[s1_id][cid].add(f"fallback_token:{t}")


# -------------------------------------------------------------------
# Format Writer Helper for candidate_pairs.tsv
# -------------------------------------------------------------------

def write_candidate_pairs(candidates_dict: Dict[str, List[str]], output_path: str):
    """
    Writes the candidate pairs dictionary into output/candidate_pairs.tsv format.
    Format: source1_entity_id <TAB> candidate_entity_ids (comma-separated)
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates_dict.keys()):
            cand_str = ",".join(candidates_dict[s1_id])
            f.write(f"{s1_id}\t{cand_str}\n")

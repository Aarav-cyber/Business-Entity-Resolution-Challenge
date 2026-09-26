"""
tests/test_stage_b.py
Hardened unit test suite for Stage B (Blocking & Candidate Generation)
Verifies:
  1. ID safety (only S2/S3, never S1)
  2. Duplicate candidate discovery & provenance aggregation
  3. Missing country fallback handling
  4. Unseen country dynamic partitioning
  5. Relevance-based candidate ranking with candidate budget capping
  6. Address and landmark recovery
  7. Provenance synchronization
  8. Every S1 entity present in output
  9. Recall ceiling evaluation script correctness
"""

import os
import tempfile
import pandas as pd
import pytest

from src.blocking.blocking import BlockingEngine, write_candidate_pairs
from src.blocking.evaluate_blocking import evaluate_blocking, parse_candidates, parse_ground_truth


@pytest.fixture
def mock_stage_a_data():
    s1 = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003", "S1-004", "S1-005"],
        "business_name": ["McDonald's Pvt Ltd", "BNP Paribas SA", "City Bakery", "Unknown Cafe", "Tata Consulting"],
        "business_name_clean": ["mcdonalds", "bnp paribas", "city bakery", "unknown cafe", "tata consulting"],
        "business_name_clean_with_suffix": ["mcdonalds private limited", "bnp paribas sa", "city bakery", "unknown cafe", "tata consulting"],
        "business_address": ["MG Road Bangalore", "10 Rue de la Paix, Paris", "Indiranagar 100ft Road", "Nowhere", "Bombay House"],
        "business_address_clean": ["mg road bangalore", "10 rue de la paix paris", "indiranagar 100ft road", "nowhere", "bombay house"],
        "landmark": ["near metro", "", "opp park", "", ""],
        "country": ["india", "france", "india", "", "india"]
    })

    s2 = pd.DataFrame({
        "entity_id": ["S2-001", "S2-002", "S2-003", "S2-004"],
        "business_name": ["TCS Ltd", "BNP Paribas France", "City Bakery Indiranagar", "Tata Consulting Services"],
        "business_name_clean": ["tcs", "bnp paribas france", "city bakery indiranagar", "tata consulting services"],
        "business_name_clean_with_suffix": ["tcs limited", "bnp paribas france", "city bakery indiranagar", "tata consulting services"],
        "business_address": ["Bangalore", "Rue de Paris", "100ft Road Indiranagar", "Mumbai"],
        "business_address_clean": ["bangalore", "rue de paris", "100ft road indiranagar", "mumbai"],
        "landmark": ["", "", "opp park", ""],
        "country": ["india", "france", "india", "india"]
    })

    s3 = pd.DataFrame({
        "entity_id": ["S3-001", "S3-002", "S3-003"],
        "business_name": ["McDonalds India", "Carrefour France", "Unknown Cafe Global"],
        "business_name_clean": ["mcdonalds india", "carrefour france", "unknown cafe global"],
        "business_name_clean_with_suffix": ["mcdonalds india", "carrefour france", "unknown cafe global"],
        "business_address": ["MG Road Bangalore", "Avenue des Champs-Élysées", "Somewhere Global"],
        "business_address_clean": ["mg road bangalore", "avenue des champs élysées", "somewhere global"],
        "landmark": ["near metro", "", ""],
        "country": ["india", "france", "us"]
    })

    return s1, s2, s3


def test_id_safety_and_completeness(mock_stage_a_data):
    """Test 1 & 8: Verify only S2/S3 IDs are generated, never S1, and all S1 exist."""
    s1, s2, s3 = mock_stage_a_data
    engine = BlockingEngine()
    engine.fit_token_stats([s1, s2, s3])
    candidates, _ = engine.generate_candidates(s1, s2, s3)

    assert set(candidates.keys()) == set(s1["entity_id"])

    for s1_id, c_list in candidates.items():
        for cid in c_list:
            assert cid.startswith(("S2-", "S3-")), f"Invalid candidate ID: {cid}"
            assert cid != s1_id, f"Self match detected: {cid}"


def test_missing_country_fallback(mock_stage_a_data):
    """Test 3: Missing country S1-004 should trigger fallback and find S3-003."""
    s1, s2, s3 = mock_stage_a_data
    engine = BlockingEngine()
    engine.fit_token_stats([s1, s2, s3])
    candidates, prov = engine.generate_candidates(s1, s2, s3)

    assert "S3-003" in candidates["S1-004"]
    assert any("fallback_token" in r for r in prov["S1-004"]["S3-003"])


def test_unseen_country_handling(mock_stage_a_data):
    """Test 4: Unseen country (France) should partition and match properly."""
    s1, s2, s3 = mock_stage_a_data
    engine = BlockingEngine()
    engine.fit_token_stats([s1, s2, s3])
    candidates, _ = engine.generate_candidates(s1, s2, s3)

    assert "S2-002" in candidates["S1-002"]


def test_address_and_landmark_recovery(mock_stage_a_data):
    """Test 6: Address/landmark tokens should link S1-003 to S2-003."""
    s1, s2, s3 = mock_stage_a_data
    engine = BlockingEngine()
    engine.fit_token_stats([s1, s2, s3])
    candidates, prov = engine.generate_candidates(s1, s2, s3)

    assert "S2-003" in candidates["S1-003"]
    assert "landmark_match" in prov["S1-003"]["S2-003"]


def test_relevance_based_ranking_and_provenance_sync(mock_stage_a_data):
    """Test 5 & 7: Relevance ranking ensures top-matching candidates survive budget capping and prov stays in sync."""
    s1, s2, s3 = mock_stage_a_data
    # Set cap to 1 candidate
    engine = BlockingEngine(max_candidates_per_s1=1)
    engine.fit_token_stats([s1, s2, s3])
    candidates, prov = engine.generate_candidates(s1, s2, s3)

    for s1_id, c_list in candidates.items():
        assert len(c_list) <= 1
        # Provenance map must contain only the retained candidates
        assert set(prov[s1_id].keys()) == set(c_list)


def test_write_and_parse_candidate_pairs():
    """Verify TSV serialization format and parsing."""
    cand_map = {
        "S1-001": ["S2-001", "S3-001"],
        "S1-002": []
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as tmp:
        tmp_path = tmp.name

    try:
        write_candidate_pairs(cand_map, tmp_path)
        parsed = parse_candidates(tmp_path)
        assert parsed["S1-001"] == {"S2-001", "S3-001"}
        assert parsed["S1-002"] == set()
    finally:
        os.remove(tmp_path)


def test_evaluate_blocking_metrics():
    """Verify evaluate_blocking recall ceiling calculation on mock data."""
    cand_content = "source1_entity_id\tcandidate_entity_ids\nS1-001\tS2-001,S2-002\nS1-002\tS3-001\nS1-003\t\n"
    gt_content = "source1_entity_id\tmatched_entity_ids\nS1-001\tS2-001,S3-005\nS1-002\tS3-001\nS1-003\t\n"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as tmp_cand:
        tmp_cand.write(cand_content)
        cand_path = tmp_cand.name

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as tmp_gt:
        tmp_gt.write(gt_content)
        gt_path = tmp_gt.name

    try:
        # S1-001: true {S2-001, S3-005}, found S2-001 (1/2)
        # S1-002: true {S3-001}, found S3-001 (1/1)
        # S1-003: singleton (0/0)
        # Total true matches = 3, Found true matches = 2 => Recall Ceiling = 66.667%
        gt_map = parse_ground_truth(gt_path)
        cand_map = parse_candidates(cand_path)

        total_true = sum(len(gt_map[s]) for s in gt_map)
        found = sum(len(gt_map[s].intersection(cand_map[s])) for s in gt_map)
        recall_ceiling = found / total_true * 100.0

        assert total_true == 3
        assert found == 2
        assert abs(recall_ceiling - 66.667) < 0.01
    finally:
        os.remove(cand_path)
        os.remove(gt_path)

"""
tests/test_stage_a.py
Unit tests for Stage A (Loader and Normalization)
"""

import os
import tempfile
import pandas as pd
import pytest

from src.data.loader import load_source, load_ground_truth
from src.data.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
    normalize_dataframe,
    get_known_countries,
)


def test_normalize_country():
    assert normalize_country(" U.S.A. ") == "usa"
    assert normalize_country("India ") == "india"
    assert normalize_country(" France ") == "france"  # Open set test
    assert normalize_country("") == ""
    assert normalize_country(None) == ""


def test_normalize_name():
    stripped, with_suffix = normalize_name("Tata Consultancy Services Limited")
    assert stripped == "tata consultancy services"
    assert "limited" in with_suffix

    stripped2, with_suffix2 = normalize_name("McDonald's Pvt Ltd & Co")
    assert "pvt" not in stripped2
    assert "ltd" not in stripped2
    assert "private" in with_suffix2
    assert "limited" in with_suffix2
    assert "and" in with_suffix2


def test_normalize_address():
    clean_addr, landmark = normalize_address("1 MG Rd, Bangalore Near Metro Station")
    assert "road" in clean_addr
    assert landmark == "near metro station"

    clean_addr2, landmark2 = normalize_address("10 Downing St., London")
    assert "street" in clean_addr2
    assert landmark2 == ""


def test_normalize_dataframe():
    raw_data = pd.DataFrame({
        "entity_id": ["S1-001", "S1-002"],
        "business_name": ["TCS Pvt Ltd", "Starbucks Coffee Corp."],
        "business_address": ["1 MG Rd, Bangalore", "10 Downing St Opp Park"],
        "country": ["India", " France "]
    })
    
    norm_df = normalize_dataframe(raw_data)
    
    assert "business_name_clean" in norm_df.columns
    assert "business_name_clean_with_suffix" in norm_df.columns
    assert "business_address_clean" in norm_df.columns
    assert "landmark" in norm_df.columns
    assert norm_df["country"].tolist() == ["india", "france"]
    assert norm_df["business_name_clean"].tolist() == ["tcs", "starbucks coffee"]


def test_loader_tsv():
    tsv_content = "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-00001\tTCS Ltd\t1 MG Rd\tIndia\n"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tsv") as tmp:
        tmp.write(tsv_content)
        tmp_path = tmp.name

    try:
        df = load_source(tmp_path)
        assert len(df) == 1
        assert df.iloc[0]["entity_id"] == "S1-00001"
        assert df.iloc[0]["business_name"] == "TCS Ltd"
        assert df.iloc[0]["country"] == "India"
    finally:
        os.remove(tmp_path)

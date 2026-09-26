from pathlib import Path
from typing import Iterator

import pandas as pd


SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


def load_source(path: str | Path) -> pd.DataFrame:
    """
    Load one source TSV file.

    All values are preserved as strings.
    Missing values are represented as empty strings.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".tsv":
        raise ValueError(f"Expected a TSV file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = set(SOURCE_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df[SOURCE_COLUMNS].fillna("")


def load_source_chunks(
    path: str | Path,
    chunksize: int = 100_000,
) -> Iterator[pd.DataFrame]:
    """
    Stream a large source TSV file in chunks.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunksize,
    )


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    """
    Load train_ground_truth.tsv.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    missing_columns = set(GROUND_TRUTH_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    return df[GROUND_TRUTH_COLUMNS].fillna("")
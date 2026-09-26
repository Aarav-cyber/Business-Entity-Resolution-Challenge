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


def load_all_sources(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """
    Loads source1, source2, and source3 dataset files from a directory.
    Supports filenames like source1.tsv, train_source1.tsv, test_source1.tsv.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    sources = {}
    for source_key in ["source1", "source2", "source3"]:
        possible_names = [
            f"{source_key}.tsv",
            f"train_{source_key}.tsv",
            f"test_{source_key}.tsv",
        ]
        found_path = None
        for name in possible_names:
            path = data_dir / name
            if path.exists():
                found_path = path
                break

        if found_path:
            sources[source_key] = load_source(found_path)
        else:
            raise FileNotFoundError(f"Could not find TSV for {source_key} in {data_dir}")

    return sources
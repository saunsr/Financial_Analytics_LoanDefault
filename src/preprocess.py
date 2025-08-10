"""
preprocess.py
-------------
Normalize/clean the raw dataset, unify target as `DEFAULT`, drop junk columns,
and save a processed CSV.

Run:
    python src/preprocess.py
"""

from pathlib import Path
import re
import pandas as pd
from typing import List
from data_loader import load_config, load_raw_dataframe


def _to_snake(name: str) -> str:
    # Lowercase, replace non-alphanum with underscore, collapse repeats
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_to_snake(c) for c in df.columns]
    return df


def resolve_target(df: pd.DataFrame, target_candidates: List[str]) -> pd.DataFrame:
    df = df.copy()
    # Normalize candidates to snake-case to match normalized columns
    norm_candidates = [_to_snake(c) for c in target_candidates]
    existing = [c for c in df.columns if c in norm_candidates]
    if not existing:
        raise KeyError(
            f"None of the target candidates {target_candidates} found in columns {list(df.columns)}"
        )
    # Use the first match
    target_col = existing[0]
    # Rename to DEFAULT
    df = df.rename(columns={target_col: "DEFAULT"})
    # Coerce to numeric 0/1 if needed (handles bools/strings like 'Yes'/'No')
    if df["DEFAULT"].dtype == "bool":
        df["DEFAULT"] = df["DEFAULT"].astype(int)
    else:
        # Map common truthy/falsey strings
        mapping = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0
        }
        if df["DEFAULT"].dtype == "object":
            df["DEFAULT"] = df["DEFAULT"].astype(str).str.strip().str.lower().map(mapping).astype("Int64")
        # If still not numeric, try to coerce
        if df["DEFAULT"].isna().any() or not pd.api.types.is_integer_dtype(df["DEFAULT"]):
            df["DEFAULT"] = pd.to_numeric(df["DEFAULT"], errors="coerce")

        # Final cast to int (drop/flag NaNs if any remain)
        if df["DEFAULT"].isna().any():
            # If a few NaNs slipped in due to odd labels, drop them
            df = df.dropna(subset=["DEFAULT"])
        df["DEFAULT"] = df["DEFAULT"].astype(int)

    # Sanity: values must be 0/1
    bad = ~df["DEFAULT"].isin([0, 1])
    if bad.any():
        raise ValueError("Found target values outside {0,1} after coercion.")
    return df


def drop_harmless_ids(df: pd.DataFrame, drop_list: List[str]) -> pd.DataFrame:
    # Normalize both sides to snake-case
    drops = {_to_snake(c) for c in drop_list}
    cols = set(df.columns)
    keep = [c for c in df.columns if c not in drops]
    dropped = list(cols - set(keep))
    if dropped:
        print("Dropping columns:", dropped)
    return df[keep]


def basic_type_fixes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Example: ensure numeric columns are actually numeric
    for col in df.columns:
        if df[col].dtype == "object":
            # try numeric coercion for finance-like columns
            if col in {"bank_balance", "annual_salary", "loan_amount"}:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def preprocess():
    cfg = load_config()
    df = load_raw_dataframe(cfg)

    # 1) normalize column names
    df = normalize_columns(df)

    # 2) unify/resolve target to `DEFAULT`
    df = resolve_target(df, cfg["schema"]["target_candidates"])

    # 3) drop harmless ID columns if they exist
    df = drop_harmless_ids(df, cfg["schema"]["drop_if_present"])

    # 4) basic dtype fixes (optional)
    df = basic_type_fixes(df)

    # 5) final sanity
    if "DEFAULT" not in df.columns:
        raise KeyError("Target column DEFAULT not present after preprocessing.")
    if df["DEFAULT"].nunique() != 2:
        raise ValueError("Target DEFAULT should be binary (0/1).")

    # 6) save processed
    out_dir = Path(cfg["paths"]["processed_data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cfg["paths"]["processed_filename"]
    df.to_csv(out_path, index=False)
    print(f"Saved processed dataset to: {out_path.resolve()}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    preprocess()

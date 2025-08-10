"""
data_loader.py
--------------
Reusable loader utilities:
- read config
- find a raw CSV under data/raw
- load dataframe safely
"""

from pathlib import Path
import glob
import yaml
import pandas as pd


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Always go to project root and find config/config.yaml
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_first_csv(raw_dir: str) -> Path:
    """Return the first CSV file found in raw_dir. Raise if none."""
    candidates = sorted(glob.glob(str(Path(raw_dir) / "*.csv")))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. "
            f"Did you unzip the Kaggle dataset into that folder?"
        )
    return Path(candidates[0])


def load_raw_dataframe(config: dict) -> pd.DataFrame:
    raw_dir = config["paths"]["raw_data_dir"]
    csv_path = find_first_csv(raw_dir)
    # Try common encodings; fall back to utf-8-sig if needed
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", low_memory=False)
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw_dataframe(cfg)
    print("Loaded file from:", find_first_csv(cfg["paths"]["raw_data_dir"]))
    print("Shape:", df.shape)
    print(df.head())
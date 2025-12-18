import pandas as pd


def load_dataset_from_json(filepath: str) -> pd.DataFrame:
    return pd.read_json(filepath)

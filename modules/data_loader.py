"""
data_loader.py
Loads and caches all 8 CSV datasets from the data/ directory.
Each dataframe is returned under a descriptive key.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Maps logical key → filename
_FILES: dict[str, str] = {
    "proj1":                "proj1.csv",
    "proj1_importance":     "proj1_importance.csv",
    "proj2":                "proj2.csv",
    "proj2_age_importance": "proj2_age_importance.csv",
    "proj2_importance":     "proj2_importance.csv",
    "proj3":                "proj3.csv",
    "proj3_importance":     "proj3_importance.csv",
    "proj3_reduced":        "proj3_reduced.csv",
}


@st.cache_data(show_spinner="Loading datasets…")
def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Read all CSVs from data/ and return them as a dict of DataFrames.
    Results are cached by Streamlit so subsequent page changes are instant.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys match _FILES above (e.g. "proj1", "proj3_importance", …).
        Missing files produce an empty DataFrame and a sidebar warning.
    """
    data: dict[str, pd.DataFrame] = {}

    for key, filename in _FILES.items():
        path = DATA_DIR / filename
        if path.exists():
            try:
                data[key] = pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not parse `{filename}`: {exc}")
                data[key] = pd.DataFrame()
        else:
            st.warning(
                f"Data file not found: `{filename}`. "
                "Plots for this dataset will be empty until the file is added."
            )
            data[key] = pd.DataFrame()

    return data

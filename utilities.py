"""
utilities.py
────────────
Helpers shared across the HealthHalo back-end.

Currently provides:

    • df_quick_overview(df: pd.DataFrame) -> str
         →  returns a concise plain-text summary of a dataframe, suitable
            for feeding into an LLM as context.

Future helpers (e.g., file-name sanitisation, plotting utilities) can live here.
"""

from __future__ import annotations

import pandas as pd


def df_quick_overview(df: pd.DataFrame, head_rows: int = 5) -> str:
    """
    Produce a human-readable synopsis of a dataframe.

    The goal is to squeeze enough signal into a small prompt
    while avoiding PHI: we summarise column statistics instead
    of sending every row.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to describe.
    head_rows : int, optional
        How many of the first rows to include verbatim (default = 5).

    Returns
    -------
    str
        A multiline string with:
            • row / column counts
            • per-column numeric stats or mode for categoricals
            • the first `head_rows` rows as CSV text
    """
    lines: list[str] = []

    # Overall shape
    lines.append(f"Rows: {len(df):,}, Columns: {len(df.columns)}\n")

    # Per-column summary
    lines.append("Column summary:")
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            mean = round(desc["mean"], 2)
            mn, mx = round(desc["min"], 2), round(desc["max"], 2)
            lines.append(f"  • {col}: mean={mean}, min={mn}, max={mx}")
        else:
            top = series.mode().iloc[0] if not series.mode().empty else "N/A"
            lines.append(f"  • {col}: most common='{top}'")

    # Sample rows (helps the LLM understand layout)
    lines.append(f"\nFirst {head_rows} rows:")
    lines.append(df.head(head_rows).to_csv(index=False))

    return "\n".join(lines)

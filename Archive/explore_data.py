"""
explore_data.py — Data Inventory Script
========================================
Run this first to see exactly what's in the CDC PLACES data
before building the correlation lesson.

Usage:
    python explore_data.py

Expects at least one of these files in the same directory:
    - cdc_places_data.csv        (full CDC dataset)
    - california_health_bivariate.csv  (pre-pivoted CA data)
"""

import os
import pandas as pd
import numpy as np


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# ── 1. Explore the full CDC dataset ─────────────────────────

full_file = "cdc_places_data.csv"

if os.path.exists(full_file):
    separator("FULL CDC PLACES DATASET")

    df = pd.read_csv(full_file, low_memory=False)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # Column types and missing values
    print("COLUMNS & MISSING VALUES:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = non_null / len(df) * 100
        dtype = df[col].dtype
        print(f"  {col:<35} {str(dtype):<10} {non_null:>8,} non-null ({pct:.1f}%)")

    # Key categorical breakdowns
    separator("STATES")
    state_counts = df["StateAbbr"].value_counts()
    print(f"Number of states: {state_counts.nunique()}")
    print(f"\nTop 10 by row count:")
    print(state_counts.head(10).to_string())

    separator("YEARS")
    if "Year" in df.columns:
        print(df["Year"].value_counts().sort_index().to_string())
    else:
        print("No 'Year' column found.")

    separator("MEASURES (MeasureId)")
    measure_counts = df["MeasureId"].value_counts()
    print(f"Number of unique measures: {measure_counts.nunique()}\n")
    for measure, count in measure_counts.items():
        print(f"  {measure:<20} {count:>8,} rows")

    separator("CATEGORIES")
    print(df["Category"].value_counts().to_string())

    separator("DATA VALUE TYPES")
    print(df["DataValueTypeID"].value_counts().to_string())

    # California-specific summary
    separator("CALIFORNIA SUBSET")
    df_ca = df[df["StateAbbr"] == "CA"]
    print(f"CA rows: {len(df_ca):,}")
    print(f"CA locations (unique): {df_ca['LocationName'].nunique()}")

    sample_locations = df_ca["LocationName"].unique()[:10]
    print(f"Sample locations: {list(sample_locations)}")

    is_zcta = str(sample_locations[0]).strip().isdigit()
    print(f"Location type: {'ZCTA (ZIP code)' if is_zcta else 'County/City'}")

    print(f"\nCA measures available:")
    for measure, count in df_ca["MeasureId"].value_counts().items():
        print(f"  {measure:<20} {count:>6,} rows")

    # Data_Value stats for key measures
    separator("DATA VALUE SUMMARY — KEY CA MEASURES")
    key_measures = ["OBESITY", "LPA", "DIABETES"]
    for m in key_measures:
        subset = df_ca[df_ca["MeasureId"] == m]["Data_Value"].dropna()
        if len(subset) > 0:
            print(f"\n  {m}:")
            print(f"    Count:  {len(subset):,}")
            print(f"    Mean:   {subset.mean():.2f}%")
            print(f"    Median: {subset.median():.2f}%")
            print(f"    Std:    {subset.std():.2f}")
            print(f"    Range:  {subset.min():.1f}% – {subset.max():.1f}%")
        else:
            print(f"\n  {m}: NOT FOUND in CA data")

else:
    print(f"'{full_file}' not found — skipping full dataset exploration.\n")


# ── 2. Explore the pre-pivoted bivariate file ───────────────

biv_file = "california_health_bivariate.csv"

if os.path.exists(biv_file):
    separator("CALIFORNIA BIVARIATE DATASET (Pre-Pivoted)")

    biv = pd.read_csv(biv_file)
    print(f"Shape: {biv.shape[0]:,} rows × {biv.shape[1]} columns")
    print(f"Columns: {list(biv.columns)}\n")

    print("SUMMARY STATISTICS:")
    print(biv.describe().round(2).to_string())

    print(f"\nMissing values:")
    for col in biv.columns:
        missing = biv[col].isna().sum()
        print(f"  {col:<20} {missing} missing")

    # Quick correlation preview
    numeric_cols = biv.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        separator("CORRELATION MATRIX PREVIEW")
        print(biv[numeric_cols].corr().round(4).to_string())

else:
    print(f"'{biv_file}' not found — skipping bivariate exploration.\n")


# ── 3. Lesson Planning Summary ──────────────────────────────

separator("LESSON PLANNING — WHAT WE CAN TEACH")

print("""
Based on the data structure, here are the correlation lessons
we can build (from basic to advanced):

LESSON 1: What Is Correlation?
  - Scatter plots of LPA vs Obesity
  - Visual intuition before any math
  - Positive, negative, no correlation examples

LESSON 2: Pearson Correlation Coefficient (r)
  - Hand-calculate r for a small subset (5-10 counties)
  - Then compute with Python and compare
  - Interpret r values: strength and direction

LESSON 3: Correlation ≠ Causation
  - Diabetes vs Obesity (strong r, but which causes which?)
  - Introduce confounding variables
  - Class discussion

LESSON 4: Significance Testing (p-value)
  - Is our r statistically significant?
  - scipy.stats.pearsonr → (r, p-value)
  - What sample size does to significance

LESSON 5: Multiple Correlations & Heatmaps
  - Correlation matrix across all available measures
  - seaborn heatmap visualization
  - Identify strongest/weakest relationships

LESSON 6: Regression (if time permits)
  - From correlation to prediction
  - Simple linear regression line
  - R² and what it means
""")

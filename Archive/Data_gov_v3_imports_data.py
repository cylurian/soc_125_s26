import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import stats


# --- Local file: set this to your downloaded CSV path ---
# Download once from: https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD
# (or whichever URL works), save it locally, and point this path to it.
LOCAL_CSV = "cdc_places_data.csv"

CANDIDATE_URLS = [
    "https://data.cdc.gov/api/views/swc5-untb/rows.csv?accessType=DOWNLOAD",
    "https://data.cdc.gov/api/views/6vp6-63id/rows.csv?accessType=DOWNLOAD",
    "https://data.cdc.gov/api/views/qnzd-25i4/rows.csv?accessType=DOWNLOAD",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TARGET_MEASURES = ["OBESITY", "LPA", "DIABETES"]


def try_download() -> pd.DataFrame | None:
    """Load data from local CSV if available, otherwise download from CDC API."""

    # --- Try local file first ---
    if os.path.exists(LOCAL_CSV):
        print(f"Loading local file: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV, low_memory=False)
        print(f"  Loaded {len(df):,} rows from local file")
        return df

    # --- Fall back to API download ---
    print(f"Local file '{LOCAL_CSV}' not found. Downloading from CDC API...")
    for url in CANDIDATE_URLS:
        print(f"Trying: {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=120)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), low_memory=False)
                print(f"  Loaded {len(df):,} rows")

                # Cache locally so future runs skip the download
                df.to_csv(LOCAL_CSV, index=False)
                print(f"  Saved to '{LOCAL_CSV}' for future runs")
                return df
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
    return None


def find_measure_keys(df: pd.DataFrame, measure_col: str) -> dict:
    """Map our target measure names to whatever key the dataset uses."""
    unique_measures = df[measure_col].dropna().unique()

    candidates = {
        "obesity": ["OBESITY", "Obesity", "obesity"],
        "lpa": ["LPA", "Lpa", "lpa", "PHYACT"],
        "diabetes": ["DIABETES", "Diabetes", "diabetes"],
    }

    found = {}
    for key, options in candidates.items():
        for c in options:
            if c in unique_measures:
                found[key] = c
                break

    print(f"\nMeasure keys found: {found}")
    missing = [k for k in candidates if k not in found]
    if missing:
        print(f"WARNING: Could not find keys for: {missing}")
        print(f"All available measures: {sorted(unique_measures)}")

    return found


def plot_scatter(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    color: str,
    max_labels: int = 80,
    outlier_percentile: float = 95,
) -> dict:
    """Draw scatter on given axes. Returns regression stats.

    For small datasets (n <= max_labels), every point is labeled.
    For large datasets (n > max_labels), only outliers beyond
    `outlier_percentile` distance from the regression line are labeled.
    """
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    ax.scatter(x, y, alpha=0.6, color=color, edgecolors="white", linewidths=0.5)

    # --- Smart annotation logic ---
    n = len(x)
    if n <= max_labels:
        # Small dataset (e.g. counties): label everything
        for location, xi, yi in zip(x.index, x, y):
            ax.annotate(
                location,
                (xi, yi),
                fontsize=7,
                alpha=0.75,
                xytext=(4, 4),
                textcoords="offset points",
            )
    else:
        # Large dataset (e.g. ZCTAs): label only outliers
        predicted = intercept + slope * x
        residuals = np.abs(y - predicted)
        threshold = np.percentile(residuals, outlier_percentile)

        labeled_count = 0
        for location, xi, yi, res in zip(x.index, x, y, residuals):
            if res >= threshold:
                ax.annotate(
                    str(location),
                    (xi, yi),
                    fontsize=6,
                    alpha=0.65,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
                labeled_count += 1

        print(f"  [{title}] {n} points — labeled {labeled_count} outliers "
              f"(>{outlier_percentile}th percentile residual)")

    # --- Draw regression line ---
    x_sorted = np.sort(x)
    ax.plot(x_sorted, intercept + slope * x_sorted,
            color=color, linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"r = {r_value:.3f}")
    ax.legend(fontsize=9, loc="upper left")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    return {
        "r": r_value,
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "n": n,
    }


def fetch_california_health_data():
    df = try_download()
    if df is None:
        print("All URLs failed.")
        return

    df_ca = df[df["StateAbbr"] == "CA"].copy()
    print(f"\nCA rows: {len(df_ca):,}")
    print(f"Sample LocationName values: {df_ca['LocationName'].unique()[:6]}")

    sample = str(df_ca["LocationName"].dropna().iloc[0]).strip()
    is_zcta = sample.isdigit()
    group_col = "ZCTA" if is_zcta else "LocationName"

    if is_zcta:
        print("Detected ZCTA format.")
        df_ca["ZCTA"] = pd.to_numeric(df_ca["LocationName"], errors="coerce")
    else:
        print("Detected county format.")

    measure_keys = find_measure_keys(df_ca, "MeasureId")
    required = ["obesity", "lpa", "diabetes"]
    if not all(k in measure_keys for k in required):
        print("Error: Missing one or more required measures. Aborting.")
        return

    obesity_key = measure_keys["obesity"]
    lpa_key = measure_keys["lpa"]
    diabetes_key = measure_keys["diabetes"]

    df_filtered = df_ca[
        df_ca["MeasureId"].isin([obesity_key, lpa_key, diabetes_key])
    ].copy()

    pivot_df = df_filtered.pivot_table(
        index=group_col,
        columns="MeasureId",
        values="Data_Value",
        aggfunc="mean",
    )

    print(f"\nPivot shape: {pivot_df.shape}")
    print(pivot_df.head())

    pivot_df.to_csv("california_health_bivariate.csv")
    print("\nSaved 'california_health_bivariate.csv'")

    clean = pivot_df[[lpa_key, diabetes_key, obesity_key]].dropna()
    print(f"Clean rows for analysis: {len(clean)}")

    if len(clean) < 3:
        print("Error: Not enough data points for regression.")
        return

    x_lpa = clean[lpa_key]
    x_diabetes = clean[diabetes_key]
    y_obesity = clean[obesity_key]

    location_label = "ZIP Codes" if is_zcta else "Counties"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"Obesity in California — Bivariate Analysis\n"
        f"(CDC PLACES Data · All California {location_label})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    stats1 = plot_scatter(
        ax=ax1,
        x=x_lpa,
        y=y_obesity,
        x_label="Physical Inactivity — % of Adults (LPA)",
        y_label="Obesity Prevalence — % of Adults",
        title="Physical Inactivity vs Obesity",
        color="darkblue",
    )

    stats2 = plot_scatter(
        ax=ax2,
        x=x_diabetes,
        y=y_obesity,
        x_label="Diabetes Prevalence — % of Adults",
        y_label="Obesity Prevalence — % of Adults",
        title="Diabetes vs Obesity",
        color="darkorange",
    )

    plt.tight_layout()
    plt.savefig("california_obesity_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n--- Regression Results ---")
    for label, s in [
        ("Physical Inactivity vs Obesity", stats1),
        ("Diabetes vs Obesity", stats2),
    ]:
        p_label = "<0.001" if s["p_value"] < 0.001 else f"{s['p_value']:.4f}"
        print(f"\n  {label}")
        print(f"    n:         {s['n']}")
        print(f"    Pearson r: {s['r']:.3f}")
        print(f"    Slope:     {s['slope']:.3f}")
        print(f"    p-value:   {p_label}")


if __name__ == "__main__":
    fetch_california_health_data()
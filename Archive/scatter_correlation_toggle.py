"""
scatter_correlation_toggle.py — Interactive Correlation Coefficient Lesson
===========================================================================
Diabetes Prevalence vs Obesity Prevalence
All 58 California Counties · CDC PLACES Data (2022–2023)

Toggle between Crude (raw %) and Age-Adjusted prevalence to see
how the correlation coefficient (r) changes.

Data Source:
    CDC PLACES: Local Data for Better Health, County Data
    https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb

Citation:
    Greenlund KJ, Lu H, Wang Y, et al. PLACES: Local Data for Better Health.
    Prev Chronic Dis 2022;19:210459. DOI: https://doi.org/10.5888/pcd19.210459

Usage:
    python scatter_correlation_toggle.py

Requires:
    - cdc_places_data.csv (in same directory)
    - pandas, numpy, scipy, plotly
    
    Install plotly if needed:  pip install plotly
"""

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go


# ── Load & Prepare Data ─────────────────────────────────────

print("Loading data...")
df = pd.read_csv("cdc_places_data.csv", low_memory=False)

# Filter to California, Diabetes and Obesity only
df_ca = df[
    (df["StateAbbr"] == "CA") &
    (df["MeasureId"].isin(["DIABETES", "OBESITY"]))
].copy()

print(f"CA rows for Diabetes & Obesity: {len(df_ca):,}")


# ── Build Crude and Age-Adjusted DataFrames ──────────────────

def build_pivot(df_ca, data_type):
    """Pivot one data value type into county × measure format."""
    subset = df_ca[df_ca["DataValueTypeID"] == data_type]
    pivot = subset.pivot_table(
        index="LocationName",
        columns="MeasureId",
        values="Data_Value",
        aggfunc="mean",
    ).dropna()
    return pivot


crude_df = build_pivot(df_ca, "CrdPrv")
age_adj_df = build_pivot(df_ca, "AgeAdjPrv")

print(f"\nCrude prevalence:       {len(crude_df)} counties")
print(f"Age-adjusted prevalence: {len(age_adj_df)} counties")


# ── Calculate Correlation Coefficients ───────────────────────

def calc_r(pivot_df):
    """Calculate Pearson r between Diabetes and Obesity columns."""
    r, p = stats.pearsonr(pivot_df["DIABETES"], pivot_df["OBESITY"])
    return r, p


r_crude, p_crude = calc_r(crude_df)
r_adj, p_adj = calc_r(age_adj_df)

print(f"\n{'='*50}")
print(f"  CORRELATION COEFFICIENTS")
print(f"{'='*50}")
print(f"  Crude:        r = {r_crude:.4f}  (p = {p_crude:.2e})")
print(f"  Age-Adjusted: r = {r_adj:.4f}  (p = {p_adj:.2e})")
print(f"{'='*50}")


# ── Classify Correlation Strength ────────────────────────────

def r_strength(r):
    a = abs(r)
    if a >= 0.8:
        return "Very Strong"
    elif a >= 0.6:
        return "Strong"
    elif a >= 0.4:
        return "Moderate"
    elif a >= 0.2:
        return "Weak"
    else:
        return "Very Weak / None"


# ── Build Interactive Plotly Chart ───────────────────────────

print("\nBuilding interactive chart...")

fig = go.Figure()

# --- Trace 1: Crude Prevalence (visible by default) ---
fig.add_trace(go.Scatter(
    x=crude_df["DIABETES"],
    y=crude_df["OBESITY"],
    mode="markers+text",
    name="Crude (Raw %)",
    text=crude_df.index,              # county names
    textposition="top right",
    textfont=dict(size=8, color="#888"),
    marker=dict(
        size=10,
        color="#E07A3A",
        opacity=0.75,
        line=dict(width=1, color="white"),
    ),
    hovertemplate=(
        "<b>%{text} County</b><br>"
        "Diabetes: %{x:.1f}%<br>"
        "Obesity: %{y:.1f}%"
        "<extra></extra>"
    ),
    visible=True,
))

# --- Trace 2: Age-Adjusted Prevalence (hidden by default) ---
fig.add_trace(go.Scatter(
    x=age_adj_df["DIABETES"],
    y=age_adj_df["OBESITY"],
    mode="markers+text",
    name="Age-Adjusted",
    text=age_adj_df.index,
    textposition="top right",
    textfont=dict(size=8, color="#888"),
    marker=dict(
        size=10,
        color="#2E6B8A",
        opacity=0.75,
        line=dict(width=1, color="white"),
    ),
    hovertemplate=(
        "<b>%{text} County</b><br>"
        "Diabetes: %{x:.1f}%<br>"
        "Obesity: %{y:.1f}%"
        "<extra></extra>"
    ),
    visible=False,
))


# ── Toggle Buttons ───────────────────────────────────────────

# Annotation text for each view
crude_explanation = (
    "<b>Crude (Raw %) — What you see is what you get</b><br>"
    "This is the actual percentage of adults with each condition.<br>"
    "Some counties have older populations, which naturally raises rates."
)

adj_explanation = (
    "<b>Age-Adjusted — Apples to apples comparison</b><br>"
    "What would each county's rate be if they all had the same age mix?<br>"
    "This removes the effect of older/younger populations."
)

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.0,
            y=1.18,
            xanchor="left",
            yanchor="top",
            buttons=[
                # Button 1: Show Crude
                dict(
                    label="   Crude (Raw %)   ",
                    method="update",
                    args=[
                        {"visible": [True, False]},       # trace visibility
                        {
                            "title.text": (
                                "Diabetes vs Obesity — 58 California Counties<br>"
                                f"<span style='font-size:20px; color:#E07A3A;'>"
                                f"Correlation Coefficient:  r = {r_crude:.4f}</span>"
                                f"<span style='font-size:14px; color:#999;'>"
                                f"  ({r_strength(r_crude)} Positive Correlation)</span>"
                            ),
                            "annotations[0].text": crude_explanation,
                        },
                    ],
                ),
                # Button 2: Show Age-Adjusted
                dict(
                    label="   Age-Adjusted   ",
                    method="update",
                    args=[
                        {"visible": [False, True]},
                        {
                            "title.text": (
                                "Diabetes vs Obesity — 58 California Counties<br>"
                                f"<span style='font-size:20px; color:#2E6B8A;'>"
                                f"Correlation Coefficient:  r = {r_adj:.4f}</span>"
                                f"<span style='font-size:14px; color:#999;'>"
                                f"  ({r_strength(r_adj)} Positive Correlation)</span>"
                            ),
                            "annotations[0].text": adj_explanation,
                        },
                    ],
                ),
            ],
            font=dict(size=13),
            bgcolor="#f0f0f0",
            bordercolor="#ccc",
        )
    ],
)


# ── Layout & Styling ────────────────────────────────────────

fig.update_layout(
    title=dict(
        text=(
            "Diabetes vs Obesity — 58 California Counties<br>"
            f"<span style='font-size:20px; color:#E07A3A;'>"
            f"Correlation Coefficient:  r = {r_crude:.4f}</span>"
            f"<span style='font-size:14px; color:#999;'>"
            f"  ({r_strength(r_crude)} Positive Correlation)</span>"
        ),
        font=dict(size=18),
        x=0.5,
        xanchor="center",
    ),
    xaxis=dict(
        title=dict(text="Diabetes Prevalence — % of Adults", font=dict(size=13)),
        range=[6.5, 16.5],
        gridcolor="#eee",
        ticksuffix="%",
    ),
    yaxis=dict(
        title=dict(text="Obesity Prevalence — % of Adults", font=dict(size=13)),
        range=[14, 40],
        gridcolor="#eee",
        ticksuffix="%",
    ),
    plot_bgcolor="white",
    paper_bgcolor="white",
    width=950,
    height=700,
    margin=dict(t=160, b=120),
    showlegend=False,

    # Explanation annotation (bottom of chart)
    annotations=[
        dict(
            text=crude_explanation,
            xref="paper", yref="paper",
            x=0.5, y=-0.12,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=12, color="#555"),
            align="center",
            bgcolor="#fdf6f0",
            bordercolor="#f0d9c4",
            borderwidth=1,
            borderpad=10,
        ),
        # Source citation (very bottom)
        dict(
            text=(
                "Data: CDC PLACES — Local Data for Better Health, County Data (2022–2023 Release)  ·  "
                "Greenlund KJ et al., Prev Chronic Dis 2022;19:210459  ·  data.cdc.gov"
            ),
            xref="paper", yref="paper",
            x=0.5, y=-0.22,
            xanchor="center", yanchor="top",
            showarrow=False,
            font=dict(size=9, color="#aaa"),
        ),
    ],
)


# ── Save & Open ──────────────────────────────────────────────

output_file = "scatter_correlation_toggle.html"
fig.write_html(output_file, include_plotlyjs=True, full_html=True)
print(f"\nSaved: {output_file}")
print("Open this file in your browser to see the interactive chart.")
print("Students can hover over dots and toggle between views.")

# Also try to open in browser automatically
try:
    import webbrowser
    webbrowser.open(output_file)
except Exception:
    pass


# ── Print Classroom Summary ─────────────────────────────────

print(f"""
{'='*60}
  WHAT THIS CHART SHOWS — TEACHER NOTES
{'='*60}

WHAT ARE WE STUDYING?
  • 58 California counties
  • Each dot represents ONE county
  • X-axis: % of adults with diabetes
  • Y-axis: % of adults who are obese

HOW MANY DATA POINTS?
  • n = 58 (one per county)
  • Data from CDC PLACES, which uses the Behavioral Risk
    Factor Surveillance System (BRFSS) survey data

WHAT IS THE CORRELATION COEFFICIENT?
  • Crude:        r = {r_crude:.4f}  ({r_strength(r_crude)})
  • Age-Adjusted: r = {r_adj:.4f}  ({r_strength(r_adj)})
  • Both are strong positive correlations

WHY DOES r CHANGE?
  • Crude includes the effect of age — counties with more
    seniors naturally show higher diabetes AND obesity
  • Age-adjusted removes that effect, isolating the
    relationship that ISN'T just about age
  • The r goes UP slightly ({r_crude:.4f} → {r_adj:.4f}),
    meaning the true relationship is a bit stronger than
    the raw numbers suggest

COUNTY NAMES ARE ON THE CHART?
  • Yes — all 58 counties are labeled
  • Hover over any dot for exact percentages
  • Students can identify their own county!

KEY DISCUSSION QUESTION:
  • This r value tells us diabetes and obesity are
    ASSOCIATED — but does one CAUSE the other?
  • What other factors might cause BOTH?
""")

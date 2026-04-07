"""
scatter_diabetes_obesity.py — Classroom Scatter Plot
=====================================================
Diabetes Prevalence vs Obesity Prevalence
All 58 California Counties · CDC PLACES Data (2022–2023)

Data Source:
    CDC PLACES: Local Data for Better Health, County Data
    https://data.cdc.gov/500-Cities-Places/PLACES-Local-Data-for-Better-Health-County-Data-20/swc5-untb

Citation:
    Greenlund KJ, Lu H, Wang Y, et al. PLACES: Local Data for Better Health.
    Prev Chronic Dis 2022;19:210459. DOI: https://doi.org/10.5888/pcd19.210459

Usage:
    python scatter_diabetes_obesity.py

Requires:
    - california_health_bivariate.csv (in same directory)
    - matplotlib, numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy import stats


# ── Load & Prepare Data ─────────────────────────────────────

df = pd.read_csv("california_health_bivariate.csv")

# Drop any rows with missing values in our two columns
df = df.dropna(subset=["DIABETES", "OBESITY"])

x = df["DIABETES"]
y = df["OBESITY"]
counties = df["LocationName"]

n = len(df)
print(f"Data points: {n} California counties")
print(f"Diabetes — Mean: {x.mean():.1f}%, Range: {x.min():.1f}%–{x.max():.1f}%")
print(f"Obesity  — Mean: {y.mean():.1f}%, Range: {y.min():.1f}%–{y.max():.1f}%")


# ── Regression ───────────────────────────────────────────────

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
r_squared = r_value ** 2

print(f"\nPearson r:  {r_value:.4f}")
print(f"R²:        {r_squared:.4f}")
print(f"Slope:     {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"p-value:   {'<0.001' if p_value < 0.001 else f'{p_value:.6f}'}")
print(f"Std Error: {std_err:.4f}")


# ── Identify Outliers (for labeling emphasis) ────────────────

predicted = intercept + slope * x
residuals = np.abs(y - predicted)
residual_threshold = np.percentile(residuals, 85)  # top 15% residuals


# ── Build the Figure ─────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 10))

# Scatter points
ax.scatter(
    x, y,
    s=80,
    alpha=0.7,
    color="#E07A3A",
    edgecolors="white",
    linewidths=0.8,
    zorder=3,
)

# Regression line
x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
y_line = intercept + slope * x_line
ax.plot(
    x_line, y_line,
    color="#C0392B",
    linewidth=2,
    linestyle="--",
    alpha=0.8,
    zorder=2,
    label=f"Best Fit Line (r = {r_value:.3f})",
)

# County labels — ALL 58 counties labeled
for county, xi, yi, res in zip(counties, x, y, residuals):
    # Outliers get bold, darker labels; others are lighter
    if res >= residual_threshold:
        ax.annotate(
            county,
            (xi, yi),
            fontsize=7.5,
            fontweight="bold",
            alpha=0.9,
            color="#2C3E50",
            xytext=(6, 6),
            textcoords="offset points",
        )
    else:
        ax.annotate(
            county,
            (xi, yi),
            fontsize=6.5,
            alpha=0.6,
            color="#555555",
            xytext=(5, 4),
            textcoords="offset points",
        )


# ── Labels & Titles ─────────────────────────────────────────

ax.set_title(
    "Diabetes Prevalence vs Obesity Prevalence\n"
    "All 58 California Counties",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

ax.set_xlabel("Diabetes Prevalence — % of Adults", fontsize=12, labelpad=10)
ax.set_ylabel("Obesity Prevalence — % of Adults", fontsize=12, labelpad=10)

ax.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax.legend(fontsize=11, loc="upper left")


# ── Stats Box ────────────────────────────────────────────────

stats_text = (
    f"n = {n} counties\n"
    f"Pearson r = {r_value:.3f}\n"
    f"R² = {r_squared:.3f}\n"
    f"Slope = {slope:.3f}\n"
    f"p-value {'< 0.001' if p_value < 0.001 else f'= {p_value:.4f}'}"
)

stats_box = AnchoredText(
    stats_text,
    loc="lower right",
    prop=dict(size=10, family="monospace"),
    frameon=True,
    pad=0.5,
)
stats_box.patch.set_boxstyle("round,pad=0.5")
stats_box.patch.set_facecolor("white")
stats_box.patch.set_alpha(0.9)
stats_box.patch.set_edgecolor("#CCCCCC")
ax.add_artist(stats_box)


# ── Source Box ───────────────────────────────────────────────

source_text = (
    "Data: CDC PLACES — Local Data for Better Health (2022–2023 Release)\n"
    "County-level model-based estimates from BRFSS survey data\n"
    "Source: data.cdc.gov · Greenlund et al., Prev Chronic Dis 2022;19:210459"
)

fig.text(
    0.5, -0.02,
    source_text,
    ha="center",
    fontsize=8,
    color="#888888",
    style="italic",
)


# ── Save & Show ──────────────────────────────────────────────

plt.tight_layout()
plt.savefig(
    "scatter_diabetes_vs_obesity.png",
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
print("\nSaved: scatter_diabetes_vs_obesity.png")
plt.show()


# ── Print Summary for Class Discussion ───────────────────────

print("\n" + "=" * 60)
print("  CLASSROOM DISCUSSION NOTES")
print("=" * 60)
print(f"""
WHAT WE'RE LOOKING AT:
  • {n} data points — one for each California county
  • Each dot represents the estimated % of adults with
    diabetes (x-axis) and obesity (y-axis) in that county

THE CORRELATION:
  • Pearson r = {r_value:.3f} → strong positive correlation
  • As diabetes prevalence increases, obesity prevalence
    tends to increase as well
  • R² = {r_squared:.3f} → about {r_squared*100:.0f}% of the variation in
    obesity can be "explained" by diabetes prevalence

THE BIG QUESTION — Correlation ≠ Causation:
  • Does obesity CAUSE diabetes?
  • Does diabetes CAUSE obesity?
  • Or does something else (poverty, food access, physical
    inactivity) CAUSE both?
  • This is why r tells us about ASSOCIATION, not causation.

NOTABLE OUTLIERS (counties far from the trend line):
""")

# Print the top outlier counties
outlier_df = df.copy()
outlier_df["residual"] = residuals
outlier_df = outlier_df.sort_values("residual", ascending=False).head(8)

for _, row in outlier_df.iterrows():
    direction = "above" if row["OBESITY"] > (intercept + slope * row["DIABETES"]) else "below"
    print(f"  • {row['LocationName']}: "
          f"Diabetes={row['DIABETES']:.1f}%, Obesity={row['OBESITY']:.1f}% "
          f"({direction} trend line)")

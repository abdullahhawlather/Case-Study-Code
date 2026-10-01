"""Survey statistics, figures and tests.

Paper link (final paper numbering): Section 2.2 - Figures 5-11, Table 6 (selected Spearman
correlations), Table 7 (chi-square tests) - and the survey-implied growth g_c of Section 3.3 (Table 8).
Collapsed-category chi-square tests with Monte Carlo p-values are supplementary.
Reads the anonymised survey file if present, otherwise the original response file
(the Name column is never used).  Output: results/survey_results.json, figures/*.png
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, chi2_contingency
from sklearn.cluster import KMeans
from common import SURVEY_ANON_FILE, SURVEY_ORIG_FILE, FIG_DIR, save_json

path = SURVEY_ANON_FILE if SURVEY_ANON_FILE.exists() else SURVEY_ORIG_FILE
df = pd.read_excel(path, sheet_name=0)
df = df[[c for c in df.columns if str(c).strip().lower() != "respondent" and not str(c).strip().lower().startswith("name")]]
df.columns = ["timestamp", "category", "frequency", "volume", "brand", "price", "supplier", "sustainability",
              "price_10", "promo_influence", "promo_type", "satisfaction", "issues", "continue",
              "volume_change", "profession"]
df["profession"] = df["profession"].astype(str).str.strip()
sat = {"Very dissatisfied": 1, "Dissatisfied": 2, "Neutral": 3, "Satisfied": 4, "Very satisfied": 5}
cont = {"Extremely Unlikely": 1, "Unlikely": 2, "Neutral": 3, "Likely": 4, "Extremely Likely": 5}
df["satisfaction_num"] = df["satisfaction"].map(sat)
df["continue_num"] = df["continue"].map(cont)
short = {"Electronics": "Electronics", "MRO(Maintenance, Repair and Operations)": "MRO",
         "Office Supplies": "Office Supplies", "Packaging Materials": "Packaging", "Raw Materials": "Raw Materials"}
df["cat"] = df["category"].map(short)

R = {"n": len(df), "first_response": str(df.timestamp.min()), "last_response": str(df.timestamp.max())}
is_student = df["profession"].str.lower().str.contains("stud|stdnt|stuent|stduent")
R["students"] = int(is_student.sum())
for c in ["cat", "frequency", "volume", "price_10", "promo_influence", "promo_type", "satisfaction",
          "issues", "continue", "volume_change"]:
    R["share_" + c] = (df[c].value_counts(normalize=True) * 100).round(1).to_dict()

# ---- Spearman correlations (Figure 6)
cols = ["brand", "price", "supplier", "sustainability", "satisfaction_num", "continue_num"]
rho = df[cols].corr(method="spearman")
pval = pd.DataFrame({b: {a: (1.0 if a == b else spearmanr(df[a], df[b])[1]) for a in cols} for b in cols})
R["spearman_rho"], R["spearman_p"] = rho.round(3).to_dict(), pval.round(4).to_dict()
ann = rho.round(2).astype(str)
for a in cols:
    for b in cols:
        if a != b:
            p = pval.loc[a, b]
            ann.loc[a, b] = f"{rho.loc[a, b]:.2f}" + ("**" if p < 0.01 else "*" if p < 0.05 else "")
plt.figure(figsize=(7.5, 6))
sns.heatmap(rho, annot=ann, fmt="", cmap="RdBu_r", center=0, vmin=-0.2, vmax=1, linewidths=.5)
plt.title("Spearman correlation (n = 150); * p<0.05, ** p<0.01")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig05_spearman_heatmap.png", dpi=170); plt.close()

# ---- Mean importance by category (Figure 7) and promotion crosstab (Figure 8)
mi = df.groupby("cat")[["brand", "price", "supplier", "sustainability"]].mean()
R["mean_importance"], R["category_n"] = mi.round(2).to_dict(), df["cat"].value_counts().to_dict()
plt.figure(figsize=(7.5, 4.6))
sns.heatmap(mi, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5)
plt.ylabel(""); plt.title("Mean importance rating (1-5) by product category")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig06_mean_importance.png", dpi=170); plt.close()
ct = pd.crosstab(df["cat"], df["promo_type"], normalize="index")
R["promotion_crosstab"] = ct.round(2).to_dict()
ct.columns = [c.replace("Volume-based tier rebates (buy more, save more)", "Volume tier rebates")
               .replace("Free or subsidized transportation/shipping costs", "Free/subsidised shipping")
               .replace("Extended payment terms or credit lines", "Extended payment terms")
               .replace("Direct percentage price discounts", "Direct % discount") for c in ct.columns]
plt.figure(figsize=(8, 4.6))
sns.heatmap(ct, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
plt.ylabel(""); plt.xlabel(""); plt.xticks(rotation=20, ha="right")
plt.title("Preferred promotion type (row share) by category")
plt.tight_layout(); plt.savefig(FIG_DIR / "fig07_promotion_crosstab.png", dpi=170); plt.close()

# ---- Jittered scatterplots (Figures 9-11)
rng = np.random.default_rng(1)
palette = dict(zip(sorted(df["cat"].unique()), sns.color_palette("tab10", 5)))

def jscatter(x, y, hue, fname, title, xl, yl, pal=None):
    plt.figure(figsize=(7.2, 5.2))
    for h, sub in df.groupby(hue):
        g = sub.groupby([x, y]).size().reset_index(name="n")
        plt.scatter(g[x] + rng.uniform(-.18, .18, len(g)), g[y] + rng.uniform(-.18, .18, len(g)),
                    s=g["n"] * 28, alpha=.6, label=str(h), color=None if pal is None else pal.get(h),
                    edgecolor="white", linewidth=.4)
    plt.xlabel(xl); plt.ylabel(yl); plt.title(title)
    plt.legend(title=hue, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=.6)
    plt.tight_layout(); plt.savefig(FIG_DIR / fname, dpi=170); plt.close()

jscatter("price", "brand", "cat", "fig08_price_vs_brand.png", "Price vs brand importance (jittered; bubble size = count)",
         "Price importance (1-5)", "Brand/quality importance (1-5)", palette)
jscatter("supplier", "sustainability", "cat", "fig09_supplier_vs_sustainability.png", "Supplier reliability vs sustainability importance",
         "Supplier reliability importance (1-5)", "Sustainability importance (1-5)", palette)
jscatter("satisfaction_num", "continue_num", "issues", "fig10_satisfaction_vs_repurchase.png",
         "Satisfaction vs repurchase intention (colour = delivery/defect issues)",
         "Satisfaction (1-5)", "Likelihood to continue (1-5)")

# ---- Composite scores and k-means segments (Figure 12)
price10 = {"Will continue buying the same quantity without hesitation": 1,
           "Uncertain / Depends on market alternatives": 2,
           "Will reduce purchase volume / look for lower-cost alternatives": 4,
           "Will immediately switch to a competing supplier or product type": 5}
def promo_code(s):
    s = str(s)
    return 1 if s.startswith("Do not") else 2 if s.startswith("Slightly") else 4 if s.startswith("Moderately") else 5 if s.startswith("Strongly") else np.nan
df["price_sensitivity"] = (df["price"] + df["price_10"].map(price10)) / 2
df["promo_responsiveness"] = df["promo_influence"].map(promo_code)
X = df[["price_sensitivity", "promo_responsiveness"]].dropna()
km = KMeans(3, n_init=10, random_state=0).fit(X)
df.loc[X.index, "segment"] = km.labels_
centres = pd.DataFrame(km.cluster_centers_, columns=["price_sensitivity", "promo_responsiveness"])
centres["n"] = np.bincount(km.labels_)
R["segments"] = centres.round(2).to_dict("records")
plt.figure(figsize=(7.2, 5.2))
for s, sub in df.dropna(subset=["segment"]).groupby("segment"):
    g = sub.groupby(["price_sensitivity", "promo_responsiveness"]).size().reset_index(name="n")
    plt.scatter(g["price_sensitivity"] + rng.uniform(-.06, .06, len(g)),
                g["promo_responsiveness"] + rng.uniform(-.1, .1, len(g)), s=g["n"] * 28, alpha=.6,
                label=f"Segment {int(s) + 1} (n={len(sub)})", edgecolor="white")
plt.xlabel("Price sensitivity (composite, 1-5)"); plt.ylabel("Promotion responsiveness (1-5)")
plt.title("Buyer segments (k-means, k = 3)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG_DIR / "fig11_segments.png", dpi=170); plt.close()

# ---- Chi-square tests with Monte Carlo p-values (Table 4)
def cramers_v(t):
    return float(np.sqrt(chi2_contingency(t)[0] / (t.values.sum() * (min(t.shape) - 1))))

def monte_carlo_p(t, B=20000, seed=0):
    r = np.random.default_rng(seed)
    a = np.repeat(np.arange(t.shape[0]), t.sum(1).values)
    b = np.repeat(np.arange(t.shape[1]), t.sum(0).values)
    obs, hits = chi2_contingency(t, correction=False)[0], 0
    for _ in range(B):
        tab = np.zeros(t.shape); np.add.at(tab, (a, r.permutation(b)), 1)
        tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
        hits += chi2_contingency(tab, correction=False)[0] >= obs - 1e-9
    return (hits + 1) / (B + 1)

tests = {}
def run_test(name, a, b):
    t = pd.crosstab(a, b)
    chi2, p, dof, expected = chi2_contingency(t)
    tests[name] = dict(chi2=round(chi2, 2), dof=int(dof), p=round(p, 4), cramers_v=round(cramers_v(t), 3),
                       cells_expected_below_5_pct=round(float((expected < 5).mean() * 100)),
                       monte_carlo_p=round(monte_carlo_p(t), 4))
df["satisfied"] = np.where(df["satisfaction_num"] >= 4, "Satisfied", "Not satisfied")
df["electronics"] = np.where(df["cat"] == "Electronics", "Electronics", "Other")
df["volume_up"] = df["volume_change"].str.contains("increase").map({True: "Increase", False: "No increase"})
df["likely"] = np.where(df["continue_num"] >= 4, "Likely", "Not likely")
run_test("category_x_satisfaction", df["category"], df["satisfaction"])
run_test("electronics_x_satisfied", df["electronics"], df["satisfied"])
run_test("promo_type_x_volume_increase", df["promo_type"], df["volume_up"])
run_test("issues_x_likely_to_continue", df["issues"], df["likely"])
R["chi_square_tests"] = tests

# ---- Survey-implied growth g_c (Section 3.3, Table 5)
midpoints = {"Significant increase (>20% growth)": 25, "Moderate increase (1\u201320% growth)": 10,
             "Remain stable / unchanged": 0, "Decrease": -10}
df["g"] = df["volume_change"].map(midpoints)
R["g_c_percent"] = {"all": round(float(df["g"].mean()), 2),
                    **{c: [round(float(s["g"].mean()), 2), len(s)] for c, s in df.groupby("cat")}}

# ---- Table 6: selected Spearman correlations (n = 150)
def sp(a, b):
    r, p = spearmanr(df[a], df[b]); return {"rho": round(float(r), 4), "p": round(float(p), 4)}
R["table6_selected_spearman"] = {"price_vs_brand": sp("price", "brand"),
                                 "supplier_vs_satisfaction": sp("supplier", "satisfaction_num"),
                                 "satisfaction_vs_repurchase": sp("satisfaction_num", "continue_num")}

# ---- Table 7: chi-square tests on the original (uncollapsed) variables
table7 = {}
for name, (a, b) in {"category_vs_satisfaction": ("category", "satisfaction"),
                     "promo_type_vs_volume_change": ("promo_type", "volume_change"),
                     "issues_vs_repurchase_intention": ("issues", "continue")}.items():
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(df[a], df[b]))
    table7[name] = {"chi2": round(float(chi2), 2), "p": round(float(p), 3), "dof": int(dof),
                    "significant_at_5pct": bool(p < 0.05)}
R["table7_chi_square"] = table7
print("Table 6:", R["table6_selected_spearman"]); print("Table 7:", table7)
save_json(R, "survey_results.json")

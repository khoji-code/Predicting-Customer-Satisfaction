SEED = 42                 
TARGET = "q21"            
TEST_SIZE = 0.30          
CV_FOLDS = 5              
RUN_CV = True             
N_BOOT = 500              
PERM_REPEATS = 10         
QUICK_RUN = False         


LABELS = {
    # "q1": "Check-in process",
}

DEMOGRAPHICS = ["q22", "q23", "q24", "q25", "q26"]
RECOMMENDATION_ITEMS = ["q20", "nps"]

CLASS_NAMES = {1: "Very low", 2: "Low", 3: "Medium", 4: "High", 5: "Very high"}
OUT_DIR = "article_outputs"
ZIP_NAME = "article_outputs"

import os
import sys
import json
import time
import glob
import shutil
import platform
import traceback
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, cohen_kappa_score,
                             mean_absolute_error, matthews_corrcoef, roc_auc_score,
                             roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, classification_report, make_scorer)

warnings.filterwarnings("ignore")
if QUICK_RUN:
    CV_FOLDS, N_BOOT, PERM_REPEATS = 3, 100, 3

IN_COLAB = "google.colab" in sys.modules
np.random.seed(SEED)

FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
for d in (FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

LOG_LINES, MANIFEST, TABLES, SUMMARY = [], [], OrderedDict(), OrderedDict()
_fig_n, _tab_n = [0], [0]


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def lab(c):
    return LABELS.get(c, str(c).upper())


def save_table(df, stem, desc, index=False):
    _tab_n[0] += 1
    name = f"Table{_tab_n[0]:02d}_{stem}"
    out = df.copy()
    out.to_csv(os.path.join(TAB_DIR, name + ".csv"), index=index, encoding="utf-8-sig")
    TABLES[name] = out.reset_index() if index else out
    MANIFEST.append(("table", name + ".csv", desc))
    return name


def save_fig(fig, stem, desc):
    _fig_n[0] += 1
    name = f"Fig{_fig_n[0]:02d}_{stem}.png"
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    MANIFEST.append(("figure", name, desc))
    return name


def run_step(name, fn):
    t0 = time.time()
    log("\n" + "=" * 78 + f"\n>>> {name}\n" + "=" * 78)
    try:
        fn()
        log(f"[OK] {name}  ({time.time() - t0:.1f}s)")
    except Exception as e:  
        log(f"[FAILED] {name}: {e}\n{traceback.format_exc()}")


sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
plt.rcParams.update({
    "font.family": "serif", "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False,
})
LIKERT_COLORS = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]  
MODEL_COLORS = {}


def find_input():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) and \
            sys.argv[1].lower().endswith((".xlsx", ".xls", ".csv")):
        return sys.argv[1]
    cands = [p for p in glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.csv")
             if not p.startswith(OUT_DIR)]
    if not cands and IN_COLAB:
        from google.colab import files
        print("\n>>> Please upload your Excel file (dataset.xlsx) ...")
        files.upload()
        cands = [p for p in glob.glob("*.xlsx") + glob.glob("*.xls") + glob.glob("*.csv")]
    if not cands:
        for folder in ("/mnt/user-data/uploads", "."):
            cands += glob.glob(os.path.join(folder, "*.xlsx"))
    if not cands:
        raise FileNotFoundError("No Excel file found. Upload dataset.xlsx and run again.")
    cands.sort(key=lambda p: (os.path.basename(p).lower() != "dataset.xlsx", p))
    return cands[0]


def load_data():
    path = find_input()
    log(f"Input file: {path}")
    df = pd.read_excel(path) if not path.lower().endswith(".csv") else pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")
    df[TARGET] = df[TARGET].astype(int)
    return df, path


df, input_path = load_data()
FEATURES = [c for c in df.columns if c != TARGET]
CLASSES = sorted(df[TARGET].unique())
log(f"Rows: {len(df):,}   Columns: {df.shape[1]}   Target: {TARGET}   Classes: {CLASSES}")
SUMMARY["data"] = dict(file=os.path.basename(input_path), n_rows=int(len(df)), n_columns=int(df.shape[1]),
                       target=TARGET, n_full_row_duplicates=int(df.duplicated().sum()),
                       columns=list(df.columns))

NAN_COLS = [c for c in FEATURES if df[c].isna().any()]
COMPLETE_FEATURES = [c for c in FEATURES if c not in NAN_COLS]
LIKERT5 = [c for c in df.columns if df[c].notna().any() and c not in DEMOGRAPHICS and
           sorted(df[c].dropna().unique().tolist()) == [1, 2, 3, 4, 5]]
CATEGORICAL = [c for c in FEATURES if (df[c].nunique() <= 4 or c in DEMOGRAPHICS)]
log(f"Columns with missing values (conditional items): {NAN_COLS}")
log(f"5-point items: {LIKERT5}")
log(f"Treated as grouping variables: {CATEGORICAL}")



def step_descriptives():
    rows = []
    for c in df.columns:
        s = df[c]
        v = s.dropna()
        rows.append(OrderedDict(variable=c, label=lab(c), n_valid=len(v), n_missing=int(s.isna().sum()),
                                pct_missing=round(100 * s.isna().mean(), 2), n_unique=int(v.nunique()),
                                mean=v.mean(), sd=v.std(), median=v.median(), min=v.min(), max=v.max(),
                                skewness=stats.skew(v), excess_kurtosis=stats.kurtosis(v)))
    t = pd.DataFrame(rows).round(3)
    save_table(t, "descriptive_statistics", "Descriptive statistics of all variables (valid N, missing %, mean, SD, skew, kurtosis)")

    fr = []
    for c in df.columns:
        vc = df[c].value_counts(dropna=True).sort_index()
        for v_, n_ in vc.items():
            fr.append(OrderedDict(variable=c, label=lab(c), value=v_, count=int(n_),
                                  percent_of_valid=round(100 * n_ / vc.sum(), 2)))
    save_table(pd.DataFrame(fr), "frequency_distributions", "Frequency and percentage of every response category for every variable")

    vc = df[TARGET].value_counts().sort_index()
    td = pd.DataFrame({"class": vc.index, "label": [CLASS_NAMES.get(i, i) for i in vc.index],
                       "count": vc.values, "percent": (100 * vc.values / vc.sum()).round(2)})
    save_table(td, "target_distribution", f"Distribution of the target variable ({TARGET}, overall satisfaction)")
    SUMMARY["target_distribution_percent"] = {int(k): float(v) for k, v in zip(td["class"], td["percent"])}
    SUMMARY["target_mean_sd"] = [float(df[TARGET].mean()), float(df[TARGET].std())]

    # Fig: target distribution
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([CLASS_NAMES.get(i, str(i)) for i in td["class"]], td["percent"], color=LIKERT_COLORS[:len(td)],
           edgecolor="black", linewidth=0.5)
    for i, (p, n) in enumerate(zip(td["percent"], td["count"])):
        ax.text(i, p + 0.6, f"{p:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Share of customers (%)")
    ax.set_xlabel("Overall satisfaction level")
    ax.set_ylim(0, td["percent"].max() * 1.22)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "target_distribution", "Distribution of overall customer satisfaction (5 classes)")

    items = LIKERT5
    pct = pd.DataFrame({c: df[c].value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0) * 100
                        for c in items}).T
    pct["mean"] = [df[c].mean() for c in pct.index]
    pct = pct.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.8, 0.32 * len(items) + 1.2))
    left = np.zeros(len(pct))
    for k in [1, 2, 3, 4, 5]:
        ax.barh([lab(c) for c in pct.index], pct[k], left=left, color=LIKERT_COLORS[k - 1],
                edgecolor="white", linewidth=0.4, label=str(k))
        left += pct[k].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of responses (%)")
    ax.legend(title="Rating", ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.grid(axis="y", visible=False)
    save_fig(fig, "likert_stacked_bars", "Response distribution of all 5-point items (sorted by mean)")

    cats = CATEGORICAL
    if cats:
        ncol = 3
        nrow = int(np.ceil(len(cats) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.8, 1.9 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, c in zip(axes, cats):
            vc = df[c].value_counts(normalize=True).sort_index() * 100
            ax.bar(vc.index.astype(str), vc.values, color="#4c78a8", edgecolor="black", linewidth=0.4)
            ax.set_title(lab(c), fontsize=8)
            ax.set_ylabel("%", fontsize=7)
            ax.grid(axis="x", visible=False)
        for ax in axes[len(cats):]:
            ax.axis("off")
        fig.tight_layout()
        save_fig(fig, "categorical_variable_distributions", "Distribution of categorical / demographic variables")

    if NAN_COLS:
        sl = []
        for c in NAN_COLS:
            m = df[c].isna()
            for g in FEATURES:
                if g == c or df[g].isna().any() or df[g].nunique() > 6:
                    continue
                for lvl in sorted(df[g].unique()):
                    sel = df[g] == lvl
                    p_miss_given = m[sel].mean()
                    p_lvl_given_miss = sel[m].mean()
                    if p_miss_given >= 0.98 and p_lvl_given_miss >= 0.98:
                        sl.append(OrderedDict(conditional_item=c, routing_variable=g, routing_level=lvl,
                                              pct_missing_when_level=round(100 * p_miss_given, 2),
                                              pct_of_missing_with_level=round(100 * p_lvl_given_miss, 2)))
        if sl:
            save_table(pd.DataFrame(sl), "skip_logic_detection",
                       "Detected survey routing: conditional items are missing (not applicable) when a routing question has a given answer")
            SUMMARY["skip_logic"] = sl
            log("Skip-logic detected: " + "; ".join(f"{r['conditional_item']} missing <=> {r['routing_variable']}={r['routing_level']}" for r in sl))

    if NAN_COLS:
        rows = []
        for c in NAN_COLS:
            m = df[c].isna()
            a, b = df.loc[m, TARGET], df.loc[~m, TARGET]
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            rows.append(OrderedDict(variable=c, label=lab(c), n_missing=int(m.sum()), pct_missing=round(100 * m.mean(), 2),
                                    target_mean_when_missing=a.mean(), target_mean_when_observed=b.mean(),
                                    mann_whitney_U=U, p_value=p, rank_biserial_r=2 * U / (len(a) * len(b)) - 1))
        mt = pd.DataFrame(rows).round(4)
        save_table(mt, "missingness_vs_satisfaction", "Missing-value analysis: is missingness related to satisfaction? (Mann-Whitney U)")
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        x = np.arange(len(NAN_COLS))
        for off, key, colr, nm in [(-0.18, "target_mean_when_missing", "#e45756", "Item missing"),
                                   (0.18, "target_mean_when_observed", "#4c78a8", "Item answered")]:
            ax.bar(x + off, mt[key], width=0.34, color=colr, edgecolor="black", linewidth=0.4, label=nm)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lab(c)}\n({p:.0f}% missing)" for c, p in zip(mt["variable"], mt["pct_missing"])])
        ax.set_ylabel("Mean overall satisfaction")
        ax.set_ylim(1, 5)
        ax.legend(frameon=False)
        ax.grid(axis="x", visible=False)
        save_fig(fig, "missingness_vs_satisfaction", "Mean satisfaction for customers who skipped vs answered each conditional item")


def step_correlations():
    corr = df.corr(method="spearman")
    rows = []
    for c in FEATURES:
        m = df[c].notna()
        x, y = df.loc[m, c], df.loc[m, TARGET]
        if x.nunique() < 2:
            continue
        rho, p = stats.spearmanr(x, y)
        n = int(m.sum())
        se = np.sqrt((1 + rho ** 2 / 2) / (n - 3))
        lo, hi = np.tanh(np.arctanh(rho) - 1.96 * se), np.tanh(np.arctanh(rho) + 1.96 * se)
        rows.append(OrderedDict(variable=c, label=lab(c), n=n, spearman_rho=rho, ci95_low=lo, ci95_high=hi, p_value=p))
    ct = pd.DataFrame(rows)
    ct = ct.reindex(ct["spearman_rho"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    save_table(ct.round(4), "spearman_with_target", "Spearman correlation of every variable with overall satisfaction (rho, 95% CI, p)")
    SUMMARY["top_spearman_with_target"] = ct.head(8)[["variable", "spearman_rho"]].round(3).to_dict("records")

    corr_out = corr.copy()
    save_table(corr_out.round(3), "spearman_matrix", "Full Spearman correlation matrix (pairwise-complete)", index=True)

    order = [TARGET] + ct["variable"].tolist()
    cm = corr.loc[order, order]
    mask = np.triu(np.ones_like(cm, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    sns.heatmap(cm, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, annot=True, fmt=".2f",
                annot_kws={"fontsize": 4.2}, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7, "label": "Spearman rho"}, ax=ax,
                xticklabels=[lab(c) for c in order], yticklabels=[lab(c) for c in order])
    ax.tick_params(labelsize=6)
    save_fig(fig, "spearman_heatmap", "Spearman correlation heatmap (variables ordered by |rho| with satisfaction)")

    fig, ax = plt.subplots(figsize=(4.8, 0.2 * len(ct) + 1))
    cs = ct.iloc[::-1]
    ax.barh([lab(c) for c in cs["variable"]], cs["spearman_rho"],
            color=["#2166ac" if r > 0 else "#b2182b" for r in cs["spearman_rho"]], edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Spearman rho with overall satisfaction")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "correlation_with_target", "Spearman correlation of each variable with overall satisfaction")


def step_reliability():
    items = [c for c in LIKERT5 if c != TARGET and c in COMPLETE_FEATURES]
    d = df[items]
    k = len(items)
    alpha = k / (k - 1) * (1 - d.var(ddof=1).sum() / d.sum(axis=1).var(ddof=1))
    rows = []
    tot = d.sum(axis=1)
    for c in items:
        rest = tot - d[c]
        r = np.corrcoef(d[c], rest)[0, 1]
        dd = d.drop(columns=[c])
        a2 = (k - 1) / (k - 2) * (1 - dd.var(ddof=1).sum() / dd.sum(axis=1).var(ddof=1))
        rows.append(OrderedDict(variable=c, label=lab(c), corrected_item_total_r=r, alpha_if_item_deleted=a2))
    t = pd.DataFrame(rows).round(3)
    t.loc[len(t)] = ["ALL ITEMS", f"Cronbach alpha = {alpha:.3f} (k={k})", np.nan, round(alpha, 3)]
    save_table(t, "cronbach_alpha_5pt_items", "Cronbach's alpha over all complete 5-point items (exploratory; items with negative item-total r may be reverse-coded or a different construct)")
    SUMMARY["cronbach_alpha_all_5pt_items"] = float(alpha)


def step_groups():
    rows, summ = [], []
    for c in CATEGORICAL:
        g = df[[c, TARGET]].dropna()
        groups = [x[TARGET].values for _, x in g.groupby(c)]
        for lvl, x in g.groupby(c):
            rows.append(OrderedDict(variable=c, label=lab(c), level=lvl, n=len(x), pct=round(100 * len(x) / len(g), 2),
                                    mean_satisfaction=x[TARGET].mean(), sd=x[TARGET].std(), median=x[TARGET].median()))
        if len(groups) == 2:
            U, p = stats.mannwhitneyu(groups[0], groups[1])
            eff = 2 * U / (len(groups[0]) * len(groups[1])) - 1
            summ.append(OrderedDict(variable=c, label=lab(c), test="Mann-Whitney U", statistic=U, p_value=p,
                                    effect_size_name="rank-biserial r", effect_size=eff))
        else:
            H, p = stats.kruskal(*groups)
            n = len(g)
            summ.append(OrderedDict(variable=c, label=lab(c), test="Kruskal-Wallis H", statistic=H, p_value=p,
                                    effect_size_name="epsilon-squared", effect_size=H * (n + 1) / (n ** 2 - 1)))
        rho, pr = stats.spearmanr(g[c], g[TARGET])
        summ[-1]["spearman_rho_with_target"] = rho
    s = pd.DataFrame(summ)
    
    
    order = np.argsort(s["p_value"].values)
    m = len(s)
    adj = np.empty(m)
    run = 0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * s["p_value"].values[idx])
        adj[idx] = min(1, run)
    s["p_holm"] = adj
    save_table(pd.DataFrame(rows).round(4), "satisfaction_by_group", "Mean/SD/median satisfaction by level of each categorical variable")
    save_table(s.round(5), "group_comparison_tests", "Tests of satisfaction differences across groups (Mann-Whitney / Kruskal-Wallis, Holm-adjusted p)")

    cats = CATEGORICAL
    ncol = 3
    nrow = int(np.ceil(len(cats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.8, 2.0 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cats):
        g = df[[c, TARGET]].dropna().groupby(c)[TARGET]
        mn, sd, n = g.mean(), g.std(), g.size()
        ci = 1.96 * sd / np.sqrt(n)
        ax.errorbar(mn.index.astype(str), mn.values, yerr=ci.values, fmt="o-", color="#4c78a8", capsize=2, lw=1)
        ax.set_title(lab(c), fontsize=8)
        ax.set_ylim(2.5, 4.5)
        ax.grid(axis="x", visible=False)
    for ax in axes[len(cats):]:
        ax.axis("off")
    axes[0].set_ylabel("Mean satisfaction (95% CI)")
    fig.tight_layout()
    save_fig(fig, "satisfaction_by_group", "Mean overall satisfaction (95% CI) by level of each categorical variable")




def ols(X, y, names):
    n, k = X.shape
    Xm = np.column_stack([np.ones(n), X])
    XtXi = np.linalg.inv(Xm.T @ Xm)
    b = XtXi @ Xm.T @ y
    fit = Xm @ b
    res = y - fit
    rss, tss = (res ** 2).sum(), ((y - y.mean()) ** 2).sum()
    dfr = n - k - 1
    s2 = rss / dfr
    se = np.sqrt(np.diag(XtXi) * s2)
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), dfr)
    tcrit = stats.t.ppf(0.975, dfr)
    h = ((Xm @ XtXi) * Xm).sum(axis=1)
    u2 = res ** 2 / (1 - h) ** 2
    V3 = XtXi @ (Xm.T @ (Xm * u2[:, None])) @ XtXi
    se3 = np.sqrt(np.diag(V3))
    p3 = 2 * stats.norm.sf(np.abs(b / se3))
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / dfr
    F = (r2 / k) / ((1 - r2) / dfr)
    pF = stats.f.sf(F, k, dfr)
    R = np.corrcoef(X, rowvar=False)
    vif = np.diag(np.linalg.inv(R)) if k > 1 else np.array([1.0])
    e2 = res ** 2
    bb = np.linalg.lstsq(Xm, e2, rcond=None)[0]
    r2a = 1 - ((e2 - Xm @ bb) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    bp_lm = n * r2a
    bp_p = stats.chi2.sf(bp_lm, k)
    dw = (np.diff(res) ** 2).sum() / rss
    sdx, sdy = X.std(axis=0, ddof=1), y.std(ddof=1)
    beta = b[1:] * sdx / sdy
    coef = pd.DataFrame(OrderedDict(
        variable=["(Intercept)"] + names, label=["(Intercept)"] + [lab(c) for c in names],
        B=b, SE=se, beta_std=np.r_[np.nan, beta], t=t, p_value=p,
        ci95_low=b - tcrit * se, ci95_high=b + tcrit * se,
        beta_ci95_low=np.r_[np.nan, (b[1:] - tcrit * se[1:]) * sdx / sdy],
        beta_ci95_high=np.r_[np.nan, (b[1:] + tcrit * se[1:]) * sdx / sdy],
        SE_HC3=se3, p_HC3=p3, VIF=np.r_[np.nan, vif]))
    summ = OrderedDict(N=n, k_predictors=k, R=np.sqrt(r2), R2=r2, adj_R2=adj, SE_estimate=np.sqrt(s2),
                       F=F, F_df1=k, F_df2=dfr, F_p=pF, BP_LM=bp_lm, BP_p=bp_p, Durbin_Watson=dw,
                       max_VIF=float(np.max(vif)), resid_skew=float(stats.skew(res)),
                       resid_excess_kurtosis=float(stats.kurtosis(res)))
    return coef, summ, fit, res


def independent_columns(M, tol=1e-8):
    from scipy.linalg import qr
    Mc = (M - M.mean(axis=0)) / np.where(M.std(axis=0) == 0, 1, M.std(axis=0))
    _, R, piv = qr(Mc, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    r = int((d > tol * d.max()).sum())
    return sorted(piv[:r].tolist())


def step_regression():
    y = df[TARGET].values.astype(float)
    XA = df[COMPLETE_FEATURES].values.astype(float)
    cA, sA, fitA, resA = ols(XA, y, COMPLETE_FEATURES)
    
    XB_df = df[COMPLETE_FEATURES].copy()
    for c in NAN_COLS:
        XB_df[c] = df[c].fillna(df[c].median())
    for c in NAN_COLS:
        XB_df[f"{c}_missing"] = df[c].isna().astype(float)
    keep = independent_columns(XB_df.values.astype(float))
    dropped = [c for i, c in enumerate(XB_df.columns) if i not in keep]
    XB_df = XB_df.iloc[:, keep]
    namesB = list(XB_df.columns)
    if dropped:
        log(f"Model B: dropped perfectly collinear columns: {dropped}")
    SUMMARY["regression_modelB_dropped_collinear"] = dropped
    cB, sB, fitB, resB = ols(XB_df.values.astype(float), y, namesB)
    cB["label"] = [lab(c.replace("_missing", "")) + (" [missing flag]" if c.endswith("_missing") else "")
                   if c != "(Intercept)" else c for c in cB["variable"]]

    summ = pd.DataFrame({"Model A (fully observed predictors)": sA, "Model B (+ conditional items & missing flags)": sB}).T
    save_table(summ.round(4), "regression_model_summary",
               "OLS model summary: R, R2, adjusted R2, SE, F, Breusch-Pagan, Durbin-Watson, max VIF", index=True)
    save_table(cA.round(5), "regression_coefficients_modelA",
               "OLS Model A coefficients: B, SE, standardized beta (+CI), t, p, HC3-robust SE/p, VIF")
    save_table(cB.round(5), "regression_coefficients_modelB", "OLS Model B coefficients (with conditional items)")
    SUMMARY["regression"] = dict(modelA={k: float(v) for k, v in sA.items()}, modelB={k: float(v) for k, v in sB.items()})
    top = cA[cA.variable != "(Intercept)"].assign(abs_beta=lambda d: d.beta_std.abs()).sort_values("abs_beta", ascending=False)
    SUMMARY["regression"]["modelA_top_standardized_betas"] = top.head(8)[["variable", "beta_std", "p_value"]].round(4).to_dict("records")
    log(f"Model A: R2={sA['R2']:.3f}, adj R2={sA['adj_R2']:.3f}, SE={sA['SE_estimate']:.3f}, max VIF={sA['max_VIF']:.2f}")
    log(f"Model B: R2={sB['R2']:.3f}, adj R2={sB['adj_R2']:.3f}")

    rk = top[["variable", "label", "beta_std", "abs_beta", "p_value"]].reset_index(drop=True)
    rk.insert(0, "rank", np.arange(1, len(rk) + 1))
    save_table(rk.round(4), "regression_beta_ranking", "Predictors ranked by |standardized beta| (Model A)")

    d = cA[cA.variable != "(Intercept)"].copy()
    d["abs"] = d.beta_std.abs()
    d = d.sort_values("abs")
    fig, ax = plt.subplots(figsize=(5.2, 0.22 * len(d) + 1.0))
    yy = np.arange(len(d))
    colors = ["#2166ac" if b > 0 else "#b2182b" for b in d.beta_std]
    ax.hlines(yy, d.beta_ci95_low, d.beta_ci95_high, color=colors, lw=1.4)
    ax.scatter(d.beta_std, yy, color=colors, s=18, zorder=3)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels(d.label)
    ax.set_xlabel("Standardized coefficient (beta) with 95% CI")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "regression_standardized_betas", "Standardized regression coefficients with 95% CI (Model A)")

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(y), size=min(5000, len(y)), replace=False)
    fig, axs = plt.subplots(2, 2, figsize=(6.8, 5.6))
    hb = axs[0, 0].hexbin(fitA, resA, gridsize=40, cmap="Blues", mincnt=1)
    axs[0, 0].axhline(0, color="red", lw=0.8)
    axs[0, 0].set(xlabel="Fitted values", ylabel="Residuals")
    axs[0, 0].set_title("(a) Residuals vs fitted", fontsize=8)
    (osm, osr), (sl, ic, r) = stats.probplot(resA[idx], dist="norm")
    axs[0, 1].scatter(osm, osr, s=3, color="#4c78a8")
    axs[0, 1].plot(osm, sl * np.array(osm) + ic, color="red", lw=0.8)
    axs[0, 1].set(xlabel="Theoretical quantiles", ylabel="Sample quantiles")
    axs[0, 1].set_title("(b) Normal Q-Q plot (n=5,000 sample)", fontsize=8)
    axs[1, 0].hist(resA, bins=50, density=True, color="#9ecae1", edgecolor="white")
    xs = np.linspace(resA.min(), resA.max(), 200)
    axs[1, 0].plot(xs, stats.norm.pdf(xs, resA.mean(), resA.std()), color="red", lw=0.9)
    axs[1, 0].set(xlabel="Residual", ylabel="Density")
    axs[1, 0].set_title("(c) Residual histogram", fontsize=8)
    bx = [fitA[y == k] for k in CLASSES]
    axs[1, 1].boxplot(bx, labels=[str(k) for k in CLASSES], showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor="#c6dbef"))
    axs[1, 1].set(xlabel="Observed satisfaction", ylabel="Fitted value")
    axs[1, 1].set_title("(d) Fitted vs observed", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "regression_diagnostics", "OLS diagnostics: residuals vs fitted, Q-Q, residual histogram, fitted vs observed")

    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Z = pd.DataFrame(StandardScaler().fit_transform(df[COMPLETE_FEATURES]), columns=COMPLETE_FEATURES)
        mod = OrderedModel(df[TARGET].astype(int).values, Z.values, distr="logit")
        res = mod.fit(method="bfgs", disp=False, maxiter=2000)
        k = len(COMPLETE_FEATURES)
        pr = np.asarray(res.params)[:k]
        se = np.asarray(res.bse)[:k]
        pv = np.asarray(res.pvalues)[:k]
        ot = pd.DataFrame(OrderedDict(variable=COMPLETE_FEATURES, label=[lab(c) for c in COMPLETE_FEATURES],
                                      coef_per_SD=pr, SE=se, z=pr / se, p_value=pv, odds_ratio_per_SD=np.exp(pr),
                                      OR_ci95_low=np.exp(pr - 1.96 * se), OR_ci95_high=np.exp(pr + 1.96 * se)))
        ot = ot.reindex(ot.coef_per_SD.abs().sort_values(ascending=False).index)
        save_table(ot.round(5), "ordered_logit", "Ordered logit (proportional odds) robustness check; predictors standardized (odds ratio per +1 SD)")
        counts = np.bincount(df[TARGET].astype(int).values)[1:]
        ll0 = float((counts * np.log(counts / counts.sum())).sum())
        SUMMARY["ordered_logit"] = dict(loglik=float(res.llf), mcfadden_pseudo_r2=float(1 - res.llf / ll0),
                                        top=ot.head(6)[["variable", "odds_ratio_per_SD"]].round(3).to_dict("records"))
        log(f"Ordered logit McFadden pseudo-R2 = {1 - res.llf / ll0:.3f}")
    except Exception as e:
        log(f"[note] ordered logit skipped: {e}")


def prep(scale=True, indicator=True):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=indicator))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return steps


def make_models(inner_jobs=-1):
    return OrderedDict([
        ("Random Forest", Pipeline(prep(False) + [("m", RandomForestClassifier(
            n_estimators=100, criterion="gini", random_state=SEED, n_jobs=inner_jobs))])),
        ("SVM (RBF)", Pipeline(prep(True) + [("m", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED))])),
        ("ANN (MLP)", Pipeline(prep(True) + [("m", MLPClassifier(
            hidden_layer_sizes=(30, 30), activation="relu", solver="adam", max_iter=500, random_state=SEED))])),
        # additional benchmarks (reviewers usually ask for these)
        ("Random Forest (class-balanced)", Pipeline(prep(False) + [("m", RandomForestClassifier(
            n_estimators=100, criterion="gini", class_weight="balanced", random_state=SEED, n_jobs=inner_jobs))])),
        ("Gradient Boosting (HGB)", HistGradientBoostingClassifier(max_iter=300, random_state=SEED)),
        ("Logistic Regression", Pipeline(prep(True) + [("m", LogisticRegression(max_iter=3000))])),
        ("Majority-class baseline", DummyClassifier(strategy="most_frequent")),
    ])


CORE = ["Random Forest", "SVM (RBF)", "ANN (MLP)"]


def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def ovr_auc(y_true, scores, classes):
    per = {}
    for j, k in enumerate(classes):
        yt = (np.asarray(y_true) == k).astype(int)
        per[k] = roc_auc_score(yt, scores[:, j]) if 0 < yt.sum() < len(yt) else np.nan
    return float(np.nanmean(list(per.values()))), per


def metric_block(y_true, y_pred, scores=None):
    d = OrderedDict()
    d["accuracy"] = accuracy_score(y_true, y_pred)
    d["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        d[f"precision_{avg}"], d[f"recall_{avg}"], d[f"f1_{avg}"] = p, r, f
    d["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    d["mcc"] = matthews_corrcoef(y_true, y_pred)
    d["mae"] = mean_absolute_error(y_true, y_pred)
    d["within_1_class_acc"] = float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1))
    if scores is not None:
        d["auc_macro_ovr"] = ovr_auc(y_true, scores, CLASSES)[0]
    return d


ML = {}   

def step_ml_holdout():
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    ML.update(Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte)
    log(f"Train: {len(Xtr):,}   Test: {len(Xte):,}")
    SUMMARY["split"] = dict(n_train=int(len(Xtr)), n_test=int(len(Xte)), test_size=TEST_SIZE, stratified=True, seed=SEED)

    fitted, preds, scores, rows, times = OrderedDict(), {}, {}, [], {}
    for name, mdl in make_models(-1).items():
        t0 = time.time()
        mdl.fit(Xtr, ytr)
        times[name] = time.time() - t0
        yp = mdl.predict(Xte)
        sc = get_scores(mdl, Xte)
        fitted[name], preds[name], scores[name] = mdl, yp, sc
        r = OrderedDict(model=name)
        r.update(metric_block(yte, yp, sc))
        r["train_time_s"] = times[name]
        rows.append(r)
        log(f"  {name:34s} acc={r['accuracy']:.4f}  macroF1={r['f1_macro']:.4f}  QWK={r['qwk']:.4f}  ({times[name]:.1f}s)")
    ML.update(fitted=fitted, preds=preds, scores=scores)
    hold = pd.DataFrame(rows)
    save_table(hold.round(4), "holdout_model_comparison", "Hold-out (70/30) test performance of all models")
    SUMMARY["holdout"] = {r["model"]: {k: float(v) for k, v in r.items() if k != "model"} for r in rows}

    cr_rows, cm_rows = [], []
    for name in fitted:
        rep = classification_report(yte, preds[name], output_dict=True, zero_division=0)
        for k in CLASSES:
            r = rep[str(k)]
            cr_rows.append(OrderedDict(model=name, **{"class": k, "class_label": CLASS_NAMES.get(k, k),
                                                      "precision": r["precision"], "recall": r["recall"],
                                                      "f1": r["f1-score"], "support": int(r["support"])}))
        for avg in ("macro avg", "weighted avg"):
            r = rep[avg]
            cr_rows.append(OrderedDict(model=name, **{"class": avg, "class_label": avg, "precision": r["precision"],
                                                      "recall": r["recall"], "f1": r["f1-score"], "support": int(r["support"])}))
        cm = confusion_matrix(yte, preds[name], labels=CLASSES)
        for i, ki in enumerate(CLASSES):
            cm_rows.append(OrderedDict(model=name, true_class=ki, **{f"pred_{kj}": int(cm[i, j]) for j, kj in enumerate(CLASSES)}))
    save_table(pd.DataFrame(cr_rows).round(4), "per_class_report", "Per-class precision / recall / F1 for all models (hold-out)")
    save_table(pd.DataFrame(cm_rows), "confusion_matrices", "Confusion matrices (counts) for all models (hold-out)")
    auc_rows = []
    for name in fitted:
        m, per = ovr_auc(yte, scores[name], CLASSES)
        auc_rows.append(OrderedDict(model=name, **{f"AUC_class{k}": v for k, v in per.items()}, AUC_macro=m))
    save_table(pd.DataFrame(auc_rows).round(4), "per_class_auc", "One-vs-rest ROC-AUC per class for all models (hold-out)")

    rng = np.random.RandomState(SEED)
    n = len(yte)
    yt = yte.values
    boots = {nm: {"acc": [], "f1": []} for nm in fitted}
    for _ in range(N_BOOT):
        ix = rng.randint(0, n, n)
        for nm in fitted:
            boots[nm]["acc"].append(np.mean(yt[ix] == preds[nm][ix]))
            boots[nm]["f1"].append(f1_score(yt[ix], preds[nm][ix], average="macro", zero_division=0))
    br = []
    for nm in fitted:
        a, f = np.array(boots[nm]["acc"]), np.array(boots[nm]["f1"])
        br.append(OrderedDict(model=nm, accuracy=np.mean(yt == preds[nm]), acc_ci95_low=np.percentile(a, 2.5),
                              acc_ci95_high=np.percentile(a, 97.5), f1_macro=f1_score(yt, preds[nm], average="macro"),
                              f1_ci95_low=np.percentile(f, 2.5), f1_ci95_high=np.percentile(f, 97.5)))
    save_table(pd.DataFrame(br).round(4), "holdout_bootstrap_ci", f"Bootstrap 95% CIs ({N_BOOT} resamples) of test accuracy and macro-F1")

    best = max(CORE, key=lambda m: accuracy_score(yte, preds[m]))
    ML["best"] = best
    mc = []
    for nm in fitted:
        if nm == best:
            continue
        a = preds[best] == yt
        b = preds[nm] == yt
        n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) > 0 else 0.0
        mc.append(OrderedDict(reference=best, comparator=nm, only_ref_correct=n01, only_comparator_correct=n10,
                              chi2_cc=chi2, p_value=stats.chi2.sf(chi2, 1)))
    save_table(pd.DataFrame(mc).round(5), "mcnemar_tests", f"McNemar tests of '{best}' against every other model (hold-out)")
    SUMMARY["best_core_model_holdout"] = best

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, nm in zip(axes, CORE):
        cm = confusion_matrix(yte, preds[nm], labels=CLASSES)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{100 * cmn[i, j]:.0f}%\n{cm[i, j]}", ha="center", va="center", fontsize=5.5,
                        color="white" if cmn[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(CLASSES)))
        ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES)
        ax.set_yticklabels(CLASSES)
        ax.set_xlabel("Predicted class")
        ax.set_title(f"{nm}\nacc = {100 * accuracy_score(yte, preds[nm]):.2f}%", fontsize=8)
        ax.grid(False)
    axes[0].set_ylabel("True class")
    fig.tight_layout()
    save_fig(fig, "confusion_matrices_core_models", "Row-normalised confusion matrices (cell = % of true class, count) for RF, SVM and ANN")

    nm = "Random Forest"
    cm = confusion_matrix(yte, preds[nm], labels=CLASSES)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    sns.heatmap(cmn * 100, annot=np.array([[f"{100 * cmn[i, j]:.1f}%\n(n={cm[i, j]})" for j in range(5)] for i in range(5)]),
                fmt="", cmap="Blues", cbar_kws={"label": "% of true class"}, ax=ax,
                xticklabels=[f"{k}" for k in CLASSES], yticklabels=[f"{k}" for k in CLASSES], annot_kws={"fontsize": 7})
    ax.set_xlabel("Predicted satisfaction class")
    ax.set_ylabel("True satisfaction class")
    save_fig(fig, "confusion_matrix_random_forest", "Confusion matrix of the Random Forest (hold-out)")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    cols = ["#b2182b", "#ef8a62", "#999999", "#67a9cf", "#2166ac"]
    Yb = label_binarize(yte, classes=CLASSES)
    for ax, nm in zip(axes, CORE):
        sc = scores[nm]
        for j, k in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(Yb[:, j], sc[:, j])
            ax.plot(fpr, tpr, color=cols[j], lw=1.1, label=f"class {k} (AUC={roc_auc_score(Yb[:, j], sc[:, j]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.6)
        ax.set_title(nm, fontsize=8)
        ax.set_xlabel("False positive rate")
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    save_fig(fig, "roc_curves_core_models", "One-vs-rest ROC curves per satisfaction class for RF, SVM and ANN")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    w = 0.26
    for i, nm in enumerate(CORE):
        dist = np.abs(yt - preds[nm])
        sh = [100 * np.mean(dist == d) for d in range(5)]
        ax.bar(np.arange(5) + (i - 1) * w, sh, width=w, label=nm, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Absolute error in satisfaction classes")
    ax.set_ylabel("Share of test cases (%)")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    save_fig(fig, "ordinal_error_distribution", "Distribution of absolute ordinal prediction error for the core models")


def step_ml_cv():
    if not RUN_CV:
        log("RUN_CV = False -> skipped")
        return
    X, y = df[FEATURES], df[TARGET]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "f1_macro": "f1_macro",
               "f1_weighted": "f1_weighted",
               "qwk": make_scorer(cohen_kappa_score, weights="quadratic"),
               "mae": make_scorer(mean_absolute_error)}
    folds, rows = [], []
    for name, mdl in make_models(1).items():
        t0 = time.time()
        cv = cross_validate(mdl, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        r = OrderedDict(model=name)
        for m in scoring:
            v = cv[f"test_{m}"]
            r[f"{m}_mean"], r[f"{m}_sd"] = v.mean(), v.std(ddof=1)
            for i, vv in enumerate(v):
                folds.append(OrderedDict(model=name, fold=i + 1, metric=m, value=vv))
        r["fit_time_s_mean"] = cv["fit_time"].mean()
        rows.append(r)
        log(f"  {name:34s} CV acc={r['accuracy_mean']:.4f} +/- {r['accuracy_sd']:.4f}   macroF1={r['f1_macro_mean']:.4f}   ({time.time() - t0:.0f}s)")
    cvt = pd.DataFrame(rows)
    save_table(cvt.round(4), "cv_model_comparison", f"{CV_FOLDS}-fold stratified cross-validation (mean and SD over folds) for all models")
    save_table(pd.DataFrame(folds).round(5), "cv_fold_level_results", "Fold-level cross-validation scores")
    ML["cv"] = cvt
    SUMMARY["cv"] = {"folds": CV_FOLDS, "results": {r["model"]: {k: float(v) for k, v in r.items() if k != "model"} for r in rows}}

    show = cvt[~cvt.model.str.contains("baseline")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=False)
    for ax, (m, ttl) in zip(axes, [("accuracy", "Accuracy"), ("f1_macro", "Macro F1"), ("qwk", "Quadratic weighted kappa")]):
        s = show.sort_values(f"{m}_mean")
        ax.barh(s.model, s[f"{m}_mean"], xerr=s[f"{m}_sd"], color="#4c78a8", edgecolor="black", linewidth=0.4, capsize=2)
        ax.set_xlim(max(0, s[f"{m}_mean"].min() - 0.08), min(1, s[f"{m}_mean"].max() + 0.04))
        ax.set_xlabel(ttl)
        ax.grid(axis="y", visible=False)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.tight_layout()
    save_fig(fig, "cv_model_comparison", f"{CV_FOLDS}-fold CV performance (mean ± SD) of all models")


def step_ml_importance():
    fitted, Xte, yte = ML["fitted"], ML["Xte"], ML["yte"]
    rf = fitted["Random Forest"]
    names = list(rf.named_steps["imp"].get_feature_names_out(FEATURES))
    gi = pd.DataFrame({"feature": names, "gini_importance": rf.named_steps["m"].feature_importances_})
    gi["label"] = [lab(n.replace("missingindicator_", "")) + (" [missing flag]" if n.startswith("missingindicator_") else "") for n in names]
    gi = gi.sort_values("gini_importance", ascending=False).reset_index(drop=True)
    save_table(gi.round(5), "rf_gini_importance", "Random Forest impurity (Gini) feature importance")

    pi = permutation_importance(rf, Xte, yte, n_repeats=PERM_REPEATS, scoring="accuracy", random_state=SEED, n_jobs=-1)
    pt = pd.DataFrame({"feature": FEATURES, "label": [lab(c) for c in FEATURES],
                       "perm_importance_mean": pi.importances_mean, "perm_importance_sd": pi.importances_std})
    pt = pt.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
    pt["rank"] = np.arange(1, len(pt) + 1)
    save_table(pt.round(5), "rf_permutation_importance", "Permutation importance (drop in test accuracy) of the Random Forest")
    ML["perm"] = pt
    SUMMARY["rf_permutation_top"] = pt.head(8)[["feature", "perm_importance_mean"]].round(4).to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.2 * len(FEATURES) + 1.2))
    g = gi.head(len(FEATURES)).iloc[::-1]
    axes[0].barh(g.label, g.gini_importance, color="#4c78a8", edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Gini importance")
    axes[0].set_title("(a) Impurity-based", fontsize=8)
    p = pt.iloc[::-1]
    axes[1].barh(p.label, p.perm_importance_mean, xerr=p.perm_importance_sd, color="#f58518", edgecolor="black",
                 linewidth=0.3, capsize=1.5)
    axes[1].set_xlabel("Decrease in test accuracy")
    axes[1].set_title("(b) Permutation-based", fontsize=8)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_fig(fig, "rf_feature_importance", "Random Forest feature importance: impurity-based and permutation-based")

    top = pt.head(6)["feature"].tolist()
    rng = np.random.RandomState(SEED)
    sub = Xte.iloc[rng.choice(len(Xte), size=min(3000, len(Xte)), replace=False)]
    cls = np.array(rf.classes_)
    pdrows = []
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey=True)
    for ax, f in zip(axes.ravel(), top):
        vals = sorted(df[f].dropna().unique())
        if len(vals) > 12:
            vals = np.unique(np.percentile(df[f].dropna(), np.linspace(2, 98, 12)))
        ev = []
        for v in vals:
            Z = sub.copy()
            Z[f] = v
            pr = rf.predict_proba(Z).mean(axis=0)
            ev.append(float((pr * cls).sum()))
            pdrows.append(OrderedDict(feature=f, value=v, expected_satisfaction=ev[-1],
                                      **{f"P_class{int(k)}": float(p_) for k, p_ in zip(cls, pr)}))
        ax.plot(vals, ev, "o-", color="#2166ac", lw=1.2, ms=3)
        ax.set_title(lab(f), fontsize=8)
        ax.set_xlabel("Rating")
    axes[0, 0].set_ylabel("Predicted mean satisfaction")
    axes[1, 0].set_ylabel("Predicted mean satisfaction")
    fig.tight_layout()
    save_fig(fig, "partial_dependence_top_features", "Partial dependence (RF): predicted mean satisfaction vs rating of the six most important features")
    save_table(pd.DataFrame(pdrows).round(4), "partial_dependence_values", "Partial dependence values (RF) for the six most important features")


def step_ml_dissatisfied():
    rf, Xte, yte = ML["fitted"]["Random Forest"], ML["Xte"], ML["yte"]
    rfb = ML["fitted"]["Random Forest (class-balanced)"]
    yt = yte.values
    out, curves = [], {}
    for nm, mdl in [("Random Forest", rf), ("Random Forest (class-balanced)", rfb)]:
        P = mdl.predict_proba(Xte)
        cls = list(mdl.classes_)
        p_dis = P[:, [cls.index(1), cls.index(2)]].sum(axis=1)
        p_1 = P[:, cls.index(1)]
        for target_name, ytrue, p in [("dissatisfied (class 1-2)", (yt <= 2).astype(int), p_dis),
                                      ("very dissatisfied (class 1)", (yt == 1).astype(int), p_1)]:
            auc, ap = roc_auc_score(ytrue, p), average_precision_score(ytrue, p)
            for thr in (0.2, 0.3, 0.4, 0.5):
                flag = p >= thr
                tp = int((flag & (ytrue == 1)).sum())
                prec = tp / flag.sum() if flag.sum() else np.nan
                rec = tp / ytrue.sum()
                f1 = 2 * prec * rec / (prec + rec) if prec == prec and (prec + rec) > 0 else np.nan
                out.append(OrderedDict(model=nm, outcome=target_name, threshold=thr, prevalence=ytrue.mean(),
                                       flagged_share=flag.mean(), precision=prec, recall=rec, f1=f1,
                                       roc_auc=auc, avg_precision=ap))
            curves[(nm, target_name)] = (ytrue, p)
    save_table(pd.DataFrame(out).round(4), "dissatisfaction_detection",
               "Early-warning performance for (very) dissatisfied customers at several probability thresholds (RF, hold-out)")
    ytrue, p = curves[("Random Forest", "dissatisfied (class 1-2)")]
    dd = pd.DataFrame({"y": ytrue, "p": p}).sort_values("p", ascending=False).reset_index(drop=True)
    dd["decile"] = pd.qcut(np.arange(len(dd)), 10, labels=False) + 1
    lift = dd.groupby("decile").agg(n=("y", "size"), dissatisfied=("y", "sum"), rate=("y", "mean")).reset_index()
    lift["cum_capture_pct"] = 100 * lift["dissatisfied"].cumsum() / lift["dissatisfied"].sum()
    lift["lift"] = lift["rate"] / ytrue.mean()
    save_table(lift.round(4), "dissatisfaction_decile_lift", "Decile lift table: dissatisfied customers captured when ranking by predicted risk (RF)")
    SUMMARY["dissatisfied_detection_rf_threshold_0.5"] = [o for o in out if o["model"] == "Random Forest" and o["threshold"] == 0.5]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for (nm, tn), (ytr_, p_) in curves.items():
        if nm != "Random Forest":
            continue
        fpr, tpr, _ = roc_curve(ytr_, p_)
        axes[0].plot(fpr, tpr, lw=1.2, label=f"{tn} (AUC={roc_auc_score(ytr_, p_):.3f})")
        pr_, rc_, _ = precision_recall_curve(ytr_, p_)
        axes[1].plot(rc_, pr_, lw=1.2, label=f"{tn} (AP={average_precision_score(ytr_, p_):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.6)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="(a) ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="(b) Precision-recall")
    for ax in axes[:2]:
        ax.legend(fontsize=5.5, frameon=False, loc="lower left" if ax is axes[0] else "upper right")
    axes[2].plot([0] + list(lift["decile"] * 10), [0] + list(lift["cum_capture_pct"]), "o-", color="#2166ac", ms=3)
    axes[2].plot([0, 100], [0, 100], "k--", lw=0.6)
    axes[2].set(xlabel="% of customers contacted (highest risk first)", ylabel="% of dissatisfied captured", title="(c) Cumulative capture")
    for ax in axes:
        ax.title.set_fontsize(8)
    fig.tight_layout()
    save_fig(fig, "dissatisfaction_early_warning", "Early-warning use case (RF): ROC, precision-recall and cumulative capture curve for dissatisfied customers")


def step_ml_learning_curve():
    Xtr, ytr, Xte, yte = ML["Xtr"], ML["ytr"], ML["Xte"], ML["yte"]
    rows = []
    rng = np.random.RandomState(SEED)
    fr = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0] if not QUICK_RUN else [0.1, 0.5, 1.0]
    for f in fr:
        n = int(len(Xtr) * f)
        ix = rng.choice(len(Xtr), n, replace=False) if f < 1 else np.arange(len(Xtr))
        m = make_models(-1)["Random Forest"].fit(Xtr.iloc[ix], ytr.iloc[ix])
        rows.append(OrderedDict(train_fraction=f, n_train=n, train_accuracy=accuracy_score(ytr.iloc[ix], m.predict(Xtr.iloc[ix])),
                                test_accuracy=accuracy_score(yte, m.predict(Xte)),
                                test_f1_macro=f1_score(yte, m.predict(Xte), average="macro")))
    lc = pd.DataFrame(rows)
    save_table(lc.round(4), "learning_curve_rf", "Learning curve of the Random Forest (accuracy vs training-set size)")
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(lc.n_train, lc.train_accuracy, "o-", label="Training accuracy", color="#999999", ms=3)
    ax.plot(lc.n_train, lc.test_accuracy, "o-", label="Test accuracy", color="#2166ac", ms=3)
    ax.plot(lc.n_train, lc.test_f1_macro, "s--", label="Test macro-F1", color="#f58518", ms=3)
    ax.set_xlabel("Number of training records")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    save_fig(fig, "learning_curve_rf", "Learning curve of the Random Forest")


def step_ml_ablation():
    X, y = df[FEATURES], df[TARGET]
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    demo = [c for c in DEMOGRAPHICS if c in FEATURES]
    rec = [c for c in RECOMMENDATION_ITEMS if c in FEATURES]
    variants = OrderedDict([
        ("All features (full model)", (FEATURES, True)),
        ("Without missing-value flags", (FEATURES, False)),
        ("Without conditional items (those with missing values)", ([c for c in FEATURES if c not in NAN_COLS], True)),
        (f"Without recommendation/NPS items ({', '.join(lab(c) for c in rec)})", ([c for c in FEATURES if c not in rec], True)),
        (f"Without demographic block ({', '.join(lab(c) for c in demo)})", ([c for c in FEATURES if c not in demo], True)),
        ("Without demographics AND recommendation/NPS items", ([c for c in FEATURES if c not in demo + rec], True)),
    ])
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro", "qwk": make_scorer(cohen_kappa_score, weights="quadratic")}
    rows = []
    for nm, (cols, ind) in variants.items():
        mdl = Pipeline(prep(False, ind) + [("m", RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1))])
        cv = cross_validate(mdl, X[cols], y, cv=skf, scoring=scoring, n_jobs=-1)
        rows.append(OrderedDict(variant=nm, n_features=len(cols), accuracy_mean=cv["test_accuracy"].mean(),
                                accuracy_sd=cv["test_accuracy"].std(ddof=1), f1_macro_mean=cv["test_f1_macro"].mean(),
                                f1_macro_sd=cv["test_f1_macro"].std(ddof=1), qwk_mean=cv["test_qwk"].mean(),
                                qwk_sd=cv["test_qwk"].std(ddof=1)))
        log(f"  {nm[:60]:60s} acc={rows[-1]['accuracy_mean']:.4f}")
    ab = pd.DataFrame(rows)
    ab["delta_accuracy_vs_full"] = ab.accuracy_mean - ab.accuracy_mean.iloc[0]
    save_table(ab.round(4), "ablation_random_forest", "Ablation study (RF, stratified CV): effect of removing feature blocks / missing flags")
    SUMMARY["ablation"] = ab.round(4).to_dict("records")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    s = ab.iloc[::-1]
    ax.barh(s.variant, s.accuracy_mean, xerr=s.accuracy_sd, color="#4c78a8", edgecolor="black", linewidth=0.4, capsize=2)
    ax.set_xlim(max(0, s.accuracy_mean.min() - 0.05), min(1, s.accuracy_mean.max() + 0.03))
    ax.set_xlabel("CV accuracy (mean ± SD)")
    ax.grid(axis="y", visible=False)
    save_fig(fig, "ablation_random_forest", "Ablation: Random Forest accuracy when feature blocks are removed")


def step_main_table():
    h = pd.DataFrame(SUMMARY["holdout"]).T.reset_index().rename(columns={"index": "model"})
    cols = ["model", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro", "qwk", "mae", "auc_macro_ovr"]
    t = h[cols].copy()
    t.columns = ["Model", "Accuracy (hold-out)", "Precision (w)", "Recall (w)", "F1 (w)", "F1 (macro)", "QWK", "MAE", "AUC (macro OvR)"]
    if "cv" in ML:
        cv = ML["cv"][["model", "accuracy_mean", "accuracy_sd", "f1_macro_mean", "f1_macro_sd"]]
        cv = cv.assign(**{f"Accuracy (CV)": cv.accuracy_mean.map("{:.4f}".format) + " ± " + cv.accuracy_sd.map("{:.4f}".format),
                          f"F1 macro (CV)": cv.f1_macro_mean.map("{:.4f}".format) + " ± " + cv.f1_macro_sd.map("{:.4f}".format)})
        t = t.merge(cv[["model", "Accuracy (CV)", "F1 macro (CV)"]], left_on="Model", right_on="model", how="left").drop(columns="model")
    t = t.sort_values("Accuracy (hold-out)", ascending=False)
    save_table(t.round(4), "MAIN_model_comparison_for_article", "MAIN comparison table (hold-out metrics + CV mean ± SD) - ready for the article")


def finish():
    SUMMARY["environment"] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                  scikit_learn=sklearn.__version__, seed=SEED, quick_run=QUICK_RUN,
                                  cpu_count=os.cpu_count())
    SUMMARY["labels_provided"] = bool(LABELS)

    class Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return None if np.isnan(o) else float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=2, cls=Enc, ensure_ascii=False)

    try:
        used = set()
        with pd.ExcelWriter(os.path.join(TAB_DIR, "All_Tables.xlsx"), engine="openpyxl") as xw:
            for nm, t in TABLES.items():
                sh = nm[:31]
                while sh in used:
                    sh = sh[:-1] + "_"
                used.add(sh)
                t.to_excel(xw, sheet_name=sh, index=False)
    except Exception as e:
        log(f"[note] Excel workbook not written: {e}")

    with open(os.path.join(OUT_DIR, "00_README.txt"), "w", encoding="utf-8") as f:
        f.write("OUTPUT INDEX\n============\n\n")
        for kind, fn, d in MANIFEST:
            f.write(f"[{kind.upper():6s}] {fn}\n          {d}\n")
        f.write("\nNOTES\n-----\n"
                "* Variables are shown with their codes (Q1, Q2, ...)\n"
                "* With N ~ 39,000 almost every p-value is < 0.001: interpret effect sizes, not only p-values.\n"
                "* Hold-out split: 70/30 stratified, seed 42.  Cross-validation: stratified k-fold.\n")
    with open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))

    shutil.make_archive(ZIP_NAME, "zip", OUT_DIR)
    log(f"\nDONE.  Figures: {sum(1 for m in MANIFEST if m[0] == 'figure')}   Tables: {sum(1 for m in MANIFEST if m[0] == 'table')}")
    log(f"Created: {ZIP_NAME}.zip   -> send this file to Claude")
    failed = [l for l in LOG_LINES if l.startswith("[FAILED]")]
    if failed:
        log("\nWARNING - some steps failed (details in run_log.txt):")
        for l in failed:
            log("  " + l.split("\n")[0])
    if IN_COLAB:
        try:
            from google.colab import files
            files.download(f"{ZIP_NAME}.zip")
        except Exception as e:
            log(f"(auto-download failed: {e}) -> download {ZIP_NAME}.zip from the file panel")


if __name__ == "__main__":
    T0 = time.time()
    run_step("1. Descriptive statistics, distributions, missingness", step_descriptives)
    run_step("2. Correlations", step_correlations)
    run_step("3. Reliability (Cronbach alpha)", step_reliability)
    run_step("4. Group comparisons", step_groups)
    run_step("5. Multiple linear regression + ordered logit", step_regression)
    run_step("6. ML: hold-out training and evaluation (7 models)", step_ml_holdout)
    run_step("7. ML: feature importance + partial dependence", step_ml_importance)
    run_step("8. ML: early-warning use case (dissatisfied customers)", step_ml_dissatisfied)
    run_step("9. ML: learning curve", step_ml_learning_curve)
    run_step("10. ML: cross-validation (this is the slow step)", step_ml_cv)
    run_step("11. ML: ablation study", step_ml_ablation)
    run_step("12. Main comparison table", step_main_table)
    log(f"\nTotal run time: {(time.time() - T0) / 60:.1f} min")
    finish()

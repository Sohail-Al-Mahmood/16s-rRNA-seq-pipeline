#!/usr/bin/env python3
"""
SHAP analysis of the stacking ensemble model for CRC vs. healthy classification
================================================================================
Matches the manuscript workflow:
  * 37 genus-level CLR features
  * 80/20 stratified train/test split
  * SMOTE applied only to the training set
  * Stacking ensemble: RF + SVM + LR + LightGBM + LDA
  * Logistic regression meta-learner
  * Kernel SHAP explanation of the stacking model
  * Beeswarm and global importance plots for the top 37 genera

Required input files in the working directory:
  clr_data.csv          rows = samples, columns = ASV/OTU IDs, CLR values
  metadata.csv          columns: SampleID, Group
  final.csv             columns: Feature ID, Taxon
  final_genus_list.txt  one genus per line

Outputs:
  SHAP_top37_importance_stack.csv
  SHAP_beeswarm_37_stack.tiff
  SHAP_global_37_stack.tiff
"""

import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Publication plot settings
# --------------------------------------------------------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif"]
mpl.rcParams["font.size"] = 16
mpl.rcParams["axes.titlesize"] = 20
mpl.rcParams["axes.labelsize"] = 18
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 14

FIGSIZE = (10, 11)
DPI = 600

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
clr_data = pd.read_csv("clr_data.csv", index_col=0)
meta = pd.read_csv("metadata.csv")
tax = pd.read_csv("final.csv")

with open("final_genus_list.txt", encoding="utf-8") as f:
    genus_list = [x.strip() for x in f.readlines() if x.strip()]

genus_list = list(dict.fromkeys(genus_list))

# --------------------------------------------------------------------------
# Align samples
# --------------------------------------------------------------------------
meta["SampleID"] = meta["SampleID"].astype(str)
clr_data.index = clr_data.index.astype(str)

common_samples = clr_data.index.intersection(meta["SampleID"])
if len(common_samples) == 0:
    raise ValueError("No common samples between clr_data.csv and metadata.csv.")

clr_data = clr_data.loc[common_samples]
meta = meta.set_index("SampleID").loc[common_samples]

# --------------------------------------------------------------------------
# Taxonomy and genus-level CLR matrix
# --------------------------------------------------------------------------
tax["Feature ID"] = tax["Feature ID"].astype(str)
tax["Genus"] = tax["Taxon"].astype(str).str.extract(r"g__([^;]+)")
tax["Genus"] = tax["Genus"].str.strip()

tax = tax[tax["Genus"].isin(genus_list)].copy()

common_features = clr_data.columns.intersection(tax["Feature ID"])
clr_sub = clr_data[common_features]

genus_df = pd.DataFrame(0.0, index=clr_sub.index, columns=genus_list)

for g in genus_list:
    asvs = tax.loc[tax["Genus"] == g, "Feature ID"]
    asvs = list(set(asvs).intersection(clr_sub.columns))
    if len(asvs) > 0:
        genus_df[g] = clr_sub[asvs].sum(axis=1)

X = genus_df.astype(float)

# --------------------------------------------------------------------------
# Define labels
# --------------------------------------------------------------------------
label_map = {
    "tumor": "tumor", "crc": "tumor", "cancer": "tumor",
    "normal": "normal", "healthy": "normal", "control": "normal"
}

y = (
    meta["Group"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(label_map)
)

if y.isna().any():
    bad = meta.loc[y.isna(), "Group"].unique().tolist()
    raise ValueError(f"Unrecognised Group labels: {bad}")

y = y.astype(str)

print("Samples:", X.shape[0])
print("Genera:", X.shape[1])
print("\nClass counts:")
print(y.value_counts())

# --------------------------------------------------------------------------
# Train/test split
# --------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y
)

print("\nTraining class counts before SMOTE:")
print(y_train.value_counts())

print("\nTest class counts:")
print(y_test.value_counts())

# --------------------------------------------------------------------------
# SMOTE: training set only
# --------------------------------------------------------------------------
smote = SMOTE()
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

X_train_bal = pd.DataFrame(X_train_bal, columns=X_train.columns)
y_train_bal = pd.Series(y_train_bal, name=y_train.name)

print("\nTraining class counts after SMOTE:")
print(y_train_bal.value_counts())

# --------------------------------------------------------------------------
# Safe feature names for LightGBM
# --------------------------------------------------------------------------
safe_cols = [f"F{i}" for i in range(X_train_bal.shape[1])]
safe_to_genus = dict(zip(safe_cols, X_train_bal.columns))

X_train_lgb = X_train_bal.copy()
X_test_lgb = X_test.copy()

X_train_lgb.columns = safe_cols
X_test_lgb.columns = safe_cols

# --------------------------------------------------------------------------
# Base learners
# --------------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    n_jobs=-1
)

svm = SVC(
    probability=True,
    class_weight="balanced"
)

lr = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

lda = LinearDiscriminantAnalysis()

lgbm = LGBMClassifier(
    n_estimators=500,
    n_jobs=-1,
    verbose=-1
)

# --------------------------------------------------------------------------
# Stacking ensemble: five base learners + LR meta-learner
# --------------------------------------------------------------------------
stack_model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("svm", svm),
        ("lr", lr),
        ("lda", lda),
        ("lgb", lgbm)
    ],
    final_estimator=LogisticRegression(
        max_iter=5000,
        class_weight="balanced"
    ),
    passthrough=True,
    n_jobs=-1
)

stack_model.fit(X_train_lgb, y_train_bal)

# --------------------------------------------------------------------------
# Test-set performance
# --------------------------------------------------------------------------
classes = list(stack_model.classes_)
positive_class = "tumor"

if positive_class not in classes:
    raise ValueError(f"'{positive_class}' not found in model classes: {classes}")

positive_idx = classes.index(positive_class)

y_pred = stack_model.predict(X_test_lgb)
y_prob = stack_model.predict_proba(X_test_lgb)[:, positive_idx]

y_test_bin = (y_test == positive_class).astype(int)

print("\nStacking model test performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"F1 score: {f1_score(y_test, y_pred, pos_label=positive_class):.3f}")
print(f"AUC-ROC:  {roc_auc_score(y_test_bin, y_prob):.3f}")

# --------------------------------------------------------------------------
# SHAP analysis
# --------------------------------------------------------------------------
background_size = min(50, X_train_lgb.shape[0])
sample_size = min(100, X_test_lgb.shape[0])

background = shap.sample(X_train_lgb, background_size)
X_shap = shap.sample(X_test_lgb, sample_size)

explainer = shap.KernelExplainer(
    stack_model.predict_proba,
    background
)

shap_values = explainer.shap_values(X_shap, nsamples=500)

# Normalize SHAP output across shap versions
if isinstance(shap_values, list):
    shap_values = shap_values[positive_idx]
elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, positive_idx]
    elif shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

shap_values = np.asarray(shap_values)

if shap_values.shape[0] != X_shap.shape[0]:
    shap_values = shap_values.T

print("\nSHAP values shape:", shap_values.shape)
print("X_shap shape:", X_shap.shape)

# --------------------------------------------------------------------------
# Global SHAP importance
# --------------------------------------------------------------------------
mean_abs = np.abs(shap_values).mean(axis=0)

top_n = min(37, X.shape[1])
top_idx = np.argsort(mean_abs)[-top_n:][::-1]

X_top = X_shap.iloc[:, top_idx].copy()
shap_top = shap_values[:, top_idx]

feat_names = np.array([safe_to_genus[c] for c in X_top.columns])
X_top.columns = feat_names

importance = mean_abs[top_idx]

importance_df = pd.DataFrame({
    "Genus": feat_names,
    "Mean_SHAP": importance
}).sort_values(by="Mean_SHAP", ascending=False)

importance_df.to_csv("SHAP_top37_importance_stack.csv", index=False)

print("\nTop 10 genera by mean |SHAP|:")
print(importance_df.head(10).to_string(index=False))

# --------------------------------------------------------------------------
# 1. SHAP beeswarm plot
# --------------------------------------------------------------------------
plt.figure(figsize=FIGSIZE)

shap.summary_plot(
    shap_top,
    X_top,
    feature_names=feat_names,
    max_display=top_n,
    plot_size=FIGSIZE,
    show=False
)

for collection in plt.gca().collections:
    collection.set_sizes([30])
    collection.set_alpha(0.85)

plt.gca().tick_params(axis="y", labelsize=15)
plt.gca().tick_params(axis="x", labelsize=13)

plt.title("SHAP Summary (Top 37 Genera)", fontsize=20)
plt.tight_layout()
plt.savefig(
    "SHAP_beeswarm_37_stack.tiff",
    dpi=DPI,
    facecolor="white"
)
plt.close()

# --------------------------------------------------------------------------
# 2. Global importance plot
# --------------------------------------------------------------------------
order = np.argsort(importance)

fig, ax = plt.subplots(figsize=FIGSIZE)

ax.barh(
    feat_names[order],
    importance[order],
    color="steelblue"
)

ax.set_xlabel("Mean |SHAP value|", fontsize=18)
ax.set_title("Global Feature Importance (Top 37 Genera)", fontsize=20)
ax.tick_params(axis="y", labelsize=15)
ax.tick_params(axis="x", labelsize=13)

plt.tight_layout()
plt.savefig(
    "SHAP_global_37_stack.tiff",
    dpi=DPI,
    facecolor="white"
)
plt.close()

print("\nDONE")
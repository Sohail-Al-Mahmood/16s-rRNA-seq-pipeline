#!/usr/bin/env python3
"""
CRC vs. healthy classification from genus-level CLR abundances
==============================================================
Reproduces the machine-learning part of:
  "Ensemble Learning-Based Identification of Key Bacterial Genera Associated
   with Colorectal Cancer: A Multi-Cohort 16S rRNA-Seq Analysis"

What this script does (and matches the manuscript, Sections 2.7-2.9)
  * 10 repeated stratified 80/20 train/test splits (seeds 123-132)
  * GridSearchCV with 10-fold stratified CV, full grids of Table 2
  * five classifiers: RF, SVM, LR, LightGBM, LDA
  * stacking ensemble: the five tuned classifiers as base learners,
    logistic regression as meta-learner
  * SMOTE applied ONLY to training data, and inside every CV training fold
    (imblearn Pipeline), so no synthetic sample ever reaches a validation fold
    or the held-out test set
  * metrics on the untouched test set: accuracy, precision, recall,
    specificity, F1, AUC-ROC, average precision (mean and SD over the splits)

What it does NOT do (separate scripts in the repository)
  QIIME 2 / MMUPHin / CLR, LASSO-Boruta-RF feature selection, SHAP,
  diversity analysis, logistic regression and FDR.

Input files (in --data-dir)
  metadata.csv          columns: SampleID, Group   (Group: tumor/CRC or normal/healthy)
  final_genus_list.txt  one genus per line (the 37 selected genera)
  EITHER genus_clr.csv  rows = samples, columns = genera, CLR values   (preferred)
  OR     clr_data.csv   rows = samples, columns = ASV/OTU IDs, CLR values
         final.csv      columns: Feature ID, Taxon  (taxonomy with g__Genus)
         (in this case CLR values of the ASVs of a genus are summed)

Usage
  python crc_ml_pipeline.py --data-dir data --out-dir results      # re-run the same command to RESUME after an interruption
  python crc_ml_pipeline.py --data-dir data --out-dir results_test --quick   # smoke test
"""

import argparse
import json
import pickle
import platform
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Configuration (matches the manuscript)
# --------------------------------------------------------------------------
SEEDS = list(range(123, 133))      # 10 repeated splits
TEST_SIZE = 0.20
CV_FOLDS = 10                       # Section 2.7
SCORING = "roc_auc"                 # criterion used to pick hyperparameters
POSITIVE = "tumor"
NEGATIVE = "normal"
LABEL_MAP = {"tumor": "tumor", "crc": "tumor", "cancer": "tumor",
             "normal": "normal", "healthy": "normal", "control": "normal"}
MODEL_ORDER = ["RF", "SVM", "LR", "LGBM", "LDA", "Stacking"]


def get_grids(quick=False):
    """Hyperparameter grids = Table 2 of the manuscript."""
    if quick:  # tiny grids, only for a smoke test
        return {
            "RF": {"model__n_estimators": [50], "model__max_depth": [None, 5],
                   "model__max_features": ["sqrt"]},
            "SVM": {"model__C": [1], "model__gamma": [0.1], "model__kernel": ["rbf"]},
            "LR": {"model__C": [1], "model__penalty": ["l2"], "model__solver": ["liblinear"]},
            "LGBM": {"model__learning_rate": [0.1], "model__num_leaves": [15],
                     "model__max_depth": [3]},
            "LDA": {"model__solver": ["svd", "lsqr"]},
        }
    return {
        "RF": {"model__n_estimators": [200, 500, 1000],
               "model__max_depth": [None, 3, 5, 10],
               "model__max_features": ["sqrt", 3, 4, 6]},
        "SVM": {"model__C": [0.1, 1, 10, 100],
                "model__gamma": [0.001, 0.01, 0.1],
                "model__kernel": ["rbf"]},
        "LR": {"model__C": [0.01, 0.1, 1, 10, 100],
               "model__penalty": ["l1", "l2"],
               "model__solver": ["liblinear"]},
        "LGBM": {"model__learning_rate": [0.01, 0.05, 0.1],
                 "model__num_leaves": [15, 31, 63],
                 "model__max_depth": [3, 5, 7]},
        "LDA": {"model__solver": ["svd", "lsqr"]},
    }


def make_estimators(seed):
    # n_jobs=1 inside the models: parallelism is done once, by GridSearchCV
    return {
        "RF": RandomForestClassifier(random_state=seed, n_jobs=1),
        "SVM": SVC(probability=True, random_state=seed),
        "LR": LogisticRegression(max_iter=5000, random_state=seed),
        "LGBM": LGBMClassifier(objective="binary", random_state=seed,
                               n_jobs=1, verbosity=-1),
        "LDA": LinearDiscriminantAnalysis(),
    }


# --------------------------------------------------------------------------
# Data loading and strict checks (errors instead of silent fixes)
# --------------------------------------------------------------------------
def load_data(data_dir: Path):
    meta = pd.read_csv(data_dir / "metadata.csv", dtype={"SampleID": str})
    if not {"SampleID", "Group"}.issubset(meta.columns):
        raise ValueError("metadata.csv must contain the columns SampleID and Group")
    if meta["SampleID"].duplicated().any():
        raise ValueError("Duplicated SampleID values in metadata.csv: "
                         f"{meta.loc[meta['SampleID'].duplicated(), 'SampleID'].tolist()[:10]}")
    meta = meta.set_index("SampleID")

    genus_list = [g.strip() for g in
                  (data_dir / "final_genus_list.txt").read_text(encoding="utf-8").splitlines()
                  if g.strip()]
    if len(set(genus_list)) != len(genus_list):
        raise ValueError("final_genus_list.txt contains duplicated genera")

    genus_file = data_dir / "genus_clr.csv"
    if genus_file.exists():
        X = pd.read_csv(genus_file, index_col=0)
        X.index = X.index.astype(str)
        missing = [g for g in genus_list if g not in X.columns]
        if missing:
            raise ValueError(f"Genera missing from genus_clr.csv: {missing}")
        X = X[genus_list].copy()
        source = "genus_clr.csv"
    else:
        clr = pd.read_csv(data_dir / "clr_data.csv", index_col=0)
        clr.index = clr.index.astype(str)
        clr.columns = clr.columns.astype(str)
        tax = pd.read_csv(data_dir / "final.csv", dtype=str)
        if not {"Feature ID", "Taxon"}.issubset(tax.columns):
            raise ValueError("final.csv must contain the columns Feature ID and Taxon")
        tax["Genus"] = tax["Taxon"].str.extract(r"g__([^;]+)", expand=False).str.strip()
        X = pd.DataFrame(index=clr.index, columns=genus_list, dtype=float)
        for g in genus_list:
            ids = sorted(set(tax.loc[tax["Genus"] == g, "Feature ID"]) & set(clr.columns))
            if not ids:
                raise ValueError(f"No features found for genus '{g}' in clr_data.csv/final.csv")
            X[g] = clr[ids].sum(axis=1)   # sum of ASV-level CLR values (see README)
        source = "clr_data.csv + final.csv (ASV CLR values summed per genus)"
        print("WARNING: genus table built by summing ASV-level CLR values. "
              "Deposit the genus-level matrix as genus_clr.csv for full transparency.")

    # align samples
    not_in_meta = X.index.difference(meta.index)
    if len(not_in_meta):
        raise ValueError(f"{len(not_in_meta)} samples have no metadata, e.g. {list(not_in_meta[:5])}")
    meta = meta.loc[X.index]

    y = (meta["Group"].astype(str).str.strip().str.lower().map(LABEL_MAP))
    if y.isna().any():
        bad = meta["Group"][y.isna()].unique().tolist()
        raise ValueError(f"Unrecognised Group labels: {bad}. Expected one of {sorted(LABEL_MAP)}")
    y = y.astype(str)

    X = X.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values (NaN/Inf) in the genus table; fix them upstream "
                         "(imputing on the full data would leak information).")
    return X, y, genus_list, source


def safe_feature_names(names):
    """LightGBM rejects feature names containing JSON special characters ( [ ] { } : , \" space ... ).
    Replace every character other than letters/digits/underscore by '_' and keep names unique.
    The original genus names are kept in genus_name_map.csv and in all reported outputs."""
    safe, seen = [], {}
    for n in names:
        s_ = re.sub(r"[^0-9A-Za-z_]", "_", str(n)).strip("_") or "genus"
        if s_ in seen:
            seen[s_] += 1
            s_ = f"{s_}_{seen[s_]}"
        else:
            seen[s_] = 0
        safe.append(s_)
    return safe


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, p_tumor):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEGATIVE, POSITIVE]).ravel()
    y_bin = (np.asarray(y_true) == POSITIVE).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "auc_roc": roc_auc_score(y_bin, p_tumor),
        "average_precision": average_precision_score(y_bin, p_tumor),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def smote_for(seed, n_min_train, folds):
    """k_neighbors must be < minority-class size of the smallest *CV training fold*."""
    n_min_fold = int(n_min_train * (folds - 1) / folds)
    return SMOTE(random_state=seed, k_neighbors=max(1, min(5, n_min_fold - 1)))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 splits, 3-fold CV, tiny grids (NOT for reporting)")
    ap.add_argument("--n-jobs", default=-1, type=int)
    args = ap.parse_args()

    seeds = SEEDS[:2] if args.quick else SEEDS
    folds = 3 if args.quick else CV_FOLDS
    grids = get_grids(args.quick)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y, genus_list, source = load_data(args.data_dir)
    print(f"Samples: {len(X)} | genera: {X.shape[1]} | source: {source}")
    print(y.value_counts().to_string())

    # LightGBM-safe feature names (original names saved to genus_name_map.csv)
    safe_names = safe_feature_names(X.columns)
    pd.DataFrame({"genus": list(X.columns), "model_feature_name": safe_names}).to_csv(
        args.out_dir / "genus_name_map.csv", index=False)
    X = X.copy()
    X.columns = safe_names

    run_info = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "seeds": seeds, "test_size": TEST_SIZE, "cv_folds": folds, "scoring": SCORING,
        "quick_mode": args.quick, "n_samples": int(len(X)), "n_genera": int(X.shape[1]),
        "class_counts": y.value_counts().to_dict(), "genus_source": source, "genera": genus_list,
    }
    try:
        import imblearn, lightgbm
        run_info.update({"imbalanced-learn": imblearn.__version__, "lightgbm": lightgbm.__version__})
    except Exception:
        pass
    (args.out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))

    fpr_grid = np.linspace(0, 1, 201)
    results, preds, best_params_log = [], [], []
    roc_curves = {m: [] for m in MODEL_ORDER}

    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    signature = f"{folds}|{json.dumps(grids, sort_keys=True, default=str)}|{X.shape}|{SCORING}"

    for run_no, seed in enumerate(seeds, start=1):
        ckpt = ckpt_dir / f"split_{run_no:02d}_seed_{seed}.pkl"
        if ckpt.exists():
            saved = pickle.load(open(ckpt, "rb"))
            if saved["signature"] == signature:
                print(f"\n=== Split {run_no}/{len(seeds)} (seed {seed}): loaded from checkpoint ===")
                results.extend(saved["results"]); preds.extend(saved["preds"])
                best_params_log.extend(saved["best_params"])
                for k, v in saved["curves"].items():
                    roc_curves[k].append(v)
                continue
            print(f"Checkpoint {ckpt.name} was made with other settings; recomputing.")
        print(f"\n=== Split {run_no}/{len(seeds)} (seed {seed}) ===")
        n_res, n_pred, n_bp = len(results), len(preds), len(best_params_log)
        split_curves = {}
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        n_min = int(y_tr.value_counts().min())
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        tuned = {}
        for name, est in make_estimators(seed).items():
            pipe = ImbPipeline([("smote", smote_for(seed, n_min, folds)), ("model", est)])
            gs = GridSearchCV(pipe, grids[name], scoring=SCORING, cv=cv,
                              n_jobs=args.n_jobs, refit=True, error_score="raise")
            gs.fit(X_tr, y_tr)
            tuned[name] = gs.best_estimator_
            best_params_log.append({"split": run_no, "seed": seed, "model": name,
                                    "cv_auc": gs.best_score_, **{k: str(v) for k, v in gs.best_params_.items()}})

        # Stacking: the five *tuned* classifiers are the base learners (each keeps its own
        # SMOTE step, refitted inside StackingClassifier's internal CV), LR is the meta-learner.
        stack = StackingClassifier(
            estimators=[(n.lower(), tuned[n]) for n in ["RF", "SVM", "LR", "LGBM", "LDA"]],
            final_estimator=LogisticRegression(max_iter=5000, random_state=seed),
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
            stack_method="predict_proba", passthrough=False, n_jobs=args.n_jobs)
        stack.fit(X_tr, y_tr)
        tuned["Stacking"] = stack

        for name in MODEL_ORDER:
            model = tuned[name]
            classes = list(model.classes_)
            p = model.predict_proba(X_te)[:, classes.index(POSITIVE)]
            pred = model.predict(X_te)
            m = compute_metrics(y_te, pred, p)
            results.append({"split": run_no, "seed": seed, "model": name, **m})
            preds.extend({"split": run_no, "seed": seed, "model": name, "SampleID": sid,
                          "y_true": a, "y_pred": b, "prob_tumor": c}
                         for sid, a, b, c in zip(X_te.index, y_te, pred, p))
            fpr, tpr, _ = roc_curve((y_te == POSITIVE).astype(int), p)
            t = np.interp(fpr_grid, fpr, tpr); t[0] = 0.0
            roc_curves[name].append(t)
            split_curves[name] = t
            print(f"  {name:9s} AUC={m['auc_roc']:.3f}  acc={m['accuracy']:.3f}  "
                  f"recall={m['recall_sensitivity']:.3f}")

        pickle.dump({"signature": signature, "results": results[n_res:], "preds": preds[n_pred:],
                     "best_params": best_params_log[n_bp:], "curves": split_curves},
                    open(ckpt, "wb"))   # allows resuming after a Colab disconnect

    # ---------------- outputs ----------------
    res = pd.DataFrame(results)
    res.to_csv(args.out_dir / "per_split_metrics.csv", index=False)
    pd.DataFrame(preds).to_csv(args.out_dir / "test_predictions_all_splits.csv", index=False)
    pd.DataFrame(best_params_log).to_csv(args.out_dir / "best_hyperparameters.csv", index=False)

    cols = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1",
            "auc_roc", "average_precision"]
    summ = res.groupby("model")[cols].agg(["mean", "std"]).reindex(MODEL_ORDER)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(args.out_dir / "mean_sd_metrics.csv", index=False)

    plt.figure(figsize=(7, 6))
    mean_curves = {"fpr": fpr_grid}
    for name in MODEL_ORDER:
        mt = np.mean(roc_curves[name], axis=0); mt[-1] = 1.0
        mean_curves[name] = mt
        auc_m = res.loc[res.model == name, "auc_roc"].mean()
        auc_s = res.loc[res.model == name, "auc_roc"].std()
        plt.plot(fpr_grid, mt, label=f"{name} (AUC {auc_m:.3f} ± {auc_s:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"Mean ROC curves over {len(seeds)} repeated test splits")
    plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(args.out_dir / "mean_ROC_curves.png", dpi=300); plt.close()
    pd.DataFrame(mean_curves).to_csv(args.out_dir / "mean_roc_curves.csv", index=False)

    print("\nSummary (mean over splits):")
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).to_string())
    print(f"\nAll outputs written to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

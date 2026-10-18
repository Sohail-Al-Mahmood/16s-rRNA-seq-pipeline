#!/usr/bin/env python3
"""
CRC vs. healthy: five classifiers on the ORIGINAL (imbalanced) data - no SMOTE, no stacking
==========================================================================================
Reproduces the "imbalanced dataset" part of the manuscript (Section 2.7-2.9, Table 3 upper block,
Figure 2b): RF, SVM, LR, LightGBM and LDA trained on the original class distribution.

  * 10 repeated stratified 80/20 train/test splits (NO random seed by default; add --seeded for seeds 123-132)
  * GridSearchCV, 10-fold stratified CV, full grids of Table 2, AUC as selection criterion
  * NO resampling (no SMOTE), NO class weights, NO stacking
  * metrics on the untouched test set: accuracy, precision, recall, specificity, F1, AUC-ROC
    (+ bootstrap 95% CI of the AUC of each split), average precision; mean and SD over the splits

Self-contained: does not need imbalanced-learn.

Input files (in --data-dir): identical to crc_ml_pipeline.py
  metadata.csv (SampleID, Group), final_genus_list.txt, and genus_clr.csv
  (or clr_data.csv + final.csv; ASV-level CLR values are then summed per genus)

Usage
  python crc_ml_imbalanced.py --data-dir data --out-dir results_imbalanced
  python crc_ml_imbalanced.py --data-dir data --out-dir test_imb --quick          # smoke test
  python crc_ml_imbalanced.py --data-dir data --out-dir results_imbalanced --seeded     # optional: seeded
Re-running the same command resumes after an interruption (checkpoints).
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "lightgbm", "scikit-learn", "pandas", "numpy", "matplotlib"])
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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix, roc_curve)
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

SEEDS = list(range(123, 133))
TEST_SIZE = 0.20
CV_FOLDS = 10
SCORING = "roc_auc"
POSITIVE, NEGATIVE = "tumor", "normal"
LABEL_MAP = {"tumor": "tumor", "crc": "tumor", "cancer": "tumor",
             "normal": "normal", "healthy": "normal", "control": "normal"}
MODEL_ORDER = ["RF", "SVM", "LR", "LGBM", "LDA"]
N_BOOT = 1000


def get_grids(quick=False):
    """Table 2 of the manuscript."""
    if quick:
        return {
            "RF": {"model__n_estimators": [50], "model__max_depth": [None, 5], "model__max_features": ["sqrt"]},
            "SVM": {"model__C": [1], "model__gamma": [0.1], "model__kernel": ["rbf"]},
            "LR": {"model__C": [1], "model__penalty": ["l2"], "model__solver": ["liblinear"]},
            "LGBM": {"model__learning_rate": [0.1], "model__num_leaves": [15], "model__max_depth": [3]},
            "LDA": {"model__solver": ["svd", "lsqr"]},
        }
    return {
        "RF": {"model__n_estimators": [200, 500, 1000], "model__max_depth": [None, 3, 5, 10],
               "model__max_features": ["sqrt", 3, 4, 6]},
        "SVM": {"model__C": [0.1, 1, 10, 100], "model__gamma": [0.001, 0.01, 0.1], "model__kernel": ["rbf"]},
        "LR": {"model__C": [0.01, 0.1, 1, 10, 100], "model__penalty": ["l1", "l2"],
               "model__solver": ["liblinear"]},
        "LGBM": {"model__learning_rate": [0.01, 0.05, 0.1], "model__num_leaves": [15, 31, 63],
                 "model__max_depth": [3, 5, 7]},
        "LDA": {"model__solver": ["svd", "lsqr"]},
    }


def make_estimators(seed):
    return {
        "RF": RandomForestClassifier(random_state=seed, n_jobs=1),
        "SVM": SVC(probability=True, random_state=seed),
        "LR": LogisticRegression(max_iter=5000, random_state=seed),
        "LGBM": LGBMClassifier(objective="binary", random_state=seed, n_jobs=1, verbosity=-1),
        "LDA": LinearDiscriminantAnalysis(),
    }


def load_data(data_dir: Path):
    meta = pd.read_csv(data_dir / "metadata.csv", dtype={"SampleID": str})
    if not {"SampleID", "Group"}.issubset(meta.columns):
        raise ValueError("metadata.csv must contain the columns SampleID and Group")
    if meta["SampleID"].duplicated().any():
        raise ValueError("Duplicated SampleID values in metadata.csv")
    meta = meta.set_index("SampleID")
    genus_list = [g.strip() for g in (data_dir / "final_genus_list.txt").read_text(encoding="utf-8").splitlines()
                  if g.strip()]
    if len(set(genus_list)) != len(genus_list):
        raise ValueError("final_genus_list.txt contains duplicated genera")

    gf = data_dir / "genus_clr.csv"
    if gf.exists():
        X = pd.read_csv(gf, index_col=0); X.index = X.index.astype(str)
        missing = [g for g in genus_list if g not in X.columns]
        if missing:
            raise ValueError(f"Genera missing from genus_clr.csv: {missing}")
        X = X[genus_list].copy(); source = "genus_clr.csv"
    else:
        clr = pd.read_csv(data_dir / "clr_data.csv", index_col=0)
        clr.index = clr.index.astype(str); clr.columns = clr.columns.astype(str)
        tax = pd.read_csv(data_dir / "final.csv", dtype=str)
        if not {"Feature ID", "Taxon"}.issubset(tax.columns):
            raise ValueError("final.csv must contain the columns Feature ID and Taxon")
        tax["Genus"] = tax["Taxon"].str.extract(r"g__([^;]+)", expand=False).str.strip()
        X = pd.DataFrame(index=clr.index, columns=genus_list, dtype=float)
        for g in genus_list:
            ids = sorted(set(tax.loc[tax["Genus"] == g, "Feature ID"]) & set(clr.columns))
            if not ids:
                raise ValueError(f"No features found for genus '{g}'")
            X[g] = clr[ids].sum(axis=1)
        source = "clr_data.csv + final.csv (ASV CLR values summed per genus)"
        print("WARNING: genus table built by summing ASV-level CLR values.")

    not_in_meta = X.index.difference(meta.index)
    if len(not_in_meta):
        raise ValueError(f"{len(not_in_meta)} samples have no metadata, e.g. {list(not_in_meta[:5])}")
    meta = meta.loc[X.index]
    y = meta["Group"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    if y.isna().any():
        raise ValueError(f"Unrecognised Group labels: {meta['Group'][y.isna()].unique().tolist()}")
    y = y.astype(str)
    X = X.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Non-finite values (NaN/Inf) in the genus table; fix them upstream.")
    return X, y, genus_list, source


def safe_feature_names(names):
    """LightGBM rejects JSON special characters in feature names; originals go to genus_name_map.csv."""
    safe, seen = [], {}
    for n in names:
        s_ = re.sub(r"[^0-9A-Za-z_]", "_", str(n)).strip("_") or "genus"
        if s_ in seen:
            seen[s_] += 1; s_ = f"{s_}_{seen[s_]}"
        else:
            seen[s_] = 0
        safe.append(s_)
    return safe


def bootstrap_auc_ci(y_bin, p, rng, n_boot=N_BOOT):
    n, aucs = len(y_bin), []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if y_bin[idx].min() == y_bin[idx].max():
            continue
        aucs.append(roc_auc_score(y_bin[idx], p[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def compute_metrics(y_true, y_pred, p, rng):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[NEGATIVE, POSITIVE]).ravel()
    y_bin = (np.asarray(y_true) == POSITIVE).astype(int)
    lo, hi = bootstrap_auc_ci(y_bin, np.asarray(p), rng)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "auc_roc": roc_auc_score(y_bin, p), "auc_ci95_low": lo, "auc_ci95_high": hi,
        "average_precision": average_precision_score(y_bin, p),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".", type=Path)
    ap.add_argument("--out-dir", default="results_imbalanced", type=Path)
    ap.add_argument("--quick", action="store_true", help="smoke test only (NOT for reporting)")
    ap.add_argument("--seeded", action="store_true",
                    help="OPTIONAL: seed splits, CV and models (seeds 123-132) for exact reproducibility. "
                         "Default: no seed is set.")
    ap.add_argument("--no-seed", action="store_true", help=argparse.SUPPRESS)  # kept for compatibility (default)
    ap.add_argument("--n-jobs", default=-1, type=int)
    args = ap.parse_args()

    n_rep = 2 if args.quick else len(SEEDS)
    no_seed = not args.seeded or args.no_seed          # default: unseeded
    seeds = [None] * n_rep if no_seed else SEEDS[:n_rep]
    folds = 3 if args.quick else CV_FOLDS
    grids = get_grids(args.quick)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X, y, genus_list, source = load_data(args.data_dir)
    print(f"Samples: {len(X)} | genera: {X.shape[1]} | source: {source}")
    print(y.value_counts().to_string())
    safe = safe_feature_names(X.columns)
    pd.DataFrame({"genus": list(X.columns), "model_feature_name": safe}).to_csv(
        args.out_dir / "genus_name_map.csv", index=False)
    X = X.copy(); X.columns = safe

    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "seeds": seeds,
            "seeded": not no_seed, "test_size": TEST_SIZE, "cv_folds": folds, "scoring": SCORING,
            "resampling": "none (original imbalanced data)", "stacking": False, "quick_mode": args.quick,
            "n_samples": int(len(X)), "n_genera": int(X.shape[1]), "class_counts": y.value_counts().to_dict(),
            "genus_source": source, "genera": genus_list, "auc_ci": f"percentile bootstrap, {N_BOOT} resamples of the test set"}
    try:
        import lightgbm; info["lightgbm"] = lightgbm.__version__
    except Exception:
        pass
    (args.out_dir / "run_info.json").write_text(json.dumps(info, indent=2))

    fpr_grid = np.linspace(0, 1, 201)
    results, preds, bp_log = [], [], []
    roc_curves = {m: [] for m in MODEL_ORDER}
    ckpt_dir = args.out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    signature = f"{folds}|{json.dumps(grids, sort_keys=True, default=str)}|{X.shape}|{SCORING}|{no_seed}"

    for run_no, seed in enumerate(seeds, start=1):
        ckpt = ckpt_dir / f"split_{run_no:02d}.pkl"
        if ckpt.exists():
            saved = pickle.load(open(ckpt, "rb"))
            if saved["signature"] == signature:
                print(f"\n=== Split {run_no}/{n_rep}: loaded from checkpoint ===")
                results.extend(saved["results"]); preds.extend(saved["preds"]); bp_log.extend(saved["bp"])
                for k, v in saved["curves"].items():
                    roc_curves[k].append(v)
                continue
            print(f"Checkpoint {ckpt.name} was made with other settings; recomputing.")
        print(f"\n=== Split {run_no}/{n_rep} (seed {seed}) ===")
        n0, n1, n2 = len(results), len(preds), len(bp_log)
        curves = {}
        rng = np.random.RandomState(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

        for name, est in make_estimators(seed).items():
            gs = GridSearchCV(Pipeline([("model", est)]), grids[name], scoring=SCORING, cv=cv,
                              n_jobs=args.n_jobs, refit=True, error_score="raise")
            gs.fit(X_tr, y_tr)
            bp_log.append({"split": run_no, "seed": seed, "model": name, "cv_auc": gs.best_score_,
                           **{k: str(v) for k, v in gs.best_params_.items()}})
            model = gs.best_estimator_
            p = model.predict_proba(X_te)[:, list(model.classes_).index(POSITIVE)]
            pred = model.predict(X_te)
            m = compute_metrics(y_te, pred, p, rng)
            results.append({"split": run_no, "seed": seed, "model": name, **m})
            preds.extend({"split": run_no, "seed": seed, "model": name, "SampleID": sid,
                          "y_true": a, "y_pred": b, "prob_tumor": c}
                         for sid, a, b, c in zip(X_te.index, y_te, pred, p))
            fpr, tpr, _ = roc_curve((y_te == POSITIVE).astype(int), p)
            t = np.interp(fpr_grid, fpr, tpr); t[0] = 0.0
            roc_curves[name].append(t); curves[name] = t
            print(f"  {name:5s} AUC={m['auc_roc']:.3f} [{m['auc_ci95_low']:.3f}-{m['auc_ci95_high']:.3f}]  "
                  f"acc={m['accuracy']:.3f}  recall={m['recall_sensitivity']:.3f}")
        pickle.dump({"signature": signature, "results": results[n0:], "preds": preds[n1:],
                     "bp": bp_log[n2:], "curves": curves}, open(ckpt, "wb"))

    res = pd.DataFrame(results)
    res.to_csv(args.out_dir / "per_split_metrics_imbalanced.csv", index=False)
    pd.DataFrame(preds).to_csv(args.out_dir / "test_predictions_all_splits.csv", index=False)
    pd.DataFrame(bp_log).to_csv(args.out_dir / "best_hyperparameters.csv", index=False)
    cols = ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "auc_roc",
            "auc_ci95_low", "auc_ci95_high", "average_precision"]
    summ = res.groupby("model")[cols].agg(["mean", "std"]).reindex(MODEL_ORDER)
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(args.out_dir / "mean_sd_metrics_imbalanced.csv", index=False)

    plt.figure(figsize=(7, 6)); mean_curves = {"fpr": fpr_grid}
    for name in MODEL_ORDER:
        mt = np.mean(roc_curves[name], axis=0); mt[-1] = 1.0; mean_curves[name] = mt
        sub = res[res.model == name]
        plt.plot(fpr_grid, mt, label=f"{name} (AUC {sub.auc_roc.mean():.3f}, "
                                      f"95% CI {sub.auc_ci95_low.mean():.3f}-{sub.auc_ci95_high.mean():.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"Imbalanced data: mean ROC curves over {n_rep} test splits")
    plt.legend(loc="lower right", fontsize=8); plt.tight_layout()
    plt.savefig(args.out_dir / "mean_ROC_curves_imbalanced.png", dpi=300); plt.close()
    pd.DataFrame(mean_curves).to_csv(args.out_dir / "mean_roc_curves_imbalanced.csv", index=False)

    print("\nSummary (mean over splits):")
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).to_string())
    print(f"\nOutputs in {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()

"""Train a shot-quality model on NBA field goal attempts.

We predict P(make) from court geometry, shot type bucket, and game state.
Train on seasons 2022-2023 and test on 2024.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

NUMERIC = ["distance", "angle", "is_three", "margin", "clock", "qtr"]
SHOT_TYPES = ["dunk", "hook", "jumper", "layup", "tip"]


def featurize(df):
    out = df.copy()
    for st in SHOT_TYPES:
        out[f"is_{st}"] = (out["shot_type"] == st).astype(int)
    return out


def evaluate(name, y, p):
    return {
        "model": name,
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shots.parquet")
    ap.add_argument("--test-season", type=int, default=2024)
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    df = featurize(pd.read_parquet(args.data))
    print(f"loaded {len(df):,} shots from seasons {sorted(df.season.unique())}")

    feats = NUMERIC + [f"is_{st}" for st in SHOT_TYPES]
    train = df[df["season"] != args.test_season]
    test = df[df["season"] == args.test_season]
    print(f"  train: {len(train):,}  test: {len(test):,}")

    Xtr, ytr = train[feats].values, train["made"].values
    Xte, yte = test[feats].values, test["made"].values

    # Pull a small validation slice out of train so xgb's early stopping
    # doesn't peek at the holdout season. Seeded for reproducibility.
    rng = np.random.default_rng(11)
    val_idx = rng.choice(len(Xtr), size=int(len(Xtr) * 0.1), replace=False)
    val_mask = np.zeros(len(Xtr), dtype=bool)
    val_mask[val_idx] = True
    Xva, yva = Xtr[val_mask], ytr[val_mask]
    Xtr, ytr = Xtr[~val_mask], ytr[~val_mask]
    print(f"  carved val: {len(Xva):,}  train after carve: {len(Xtr):,}")

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    # league fg% baseline
    base = ytr.mean()
    results.append(evaluate("league_fg_pct", yte, np.full_like(yte, base, dtype=float)))

    # distance-only logistic
    Xtr_d = train[["distance"]].values
    Xte_d = test[["distance"]].values
    logit_d = LogisticRegression(max_iter=1000)
    logit_d.fit(Xtr_d, ytr)
    p_d = logit_d.predict_proba(Xte_d)[:, 1]
    results.append(evaluate("distance_only_logit", yte, p_d))

    # full logistic
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xtr_s, ytr)
    p_logit = logit.predict_proba(Xte_s)[:, 1]
    results.append(evaluate("logistic", yte, p_logit))

    # xgboost with early stopping on the carved val slice
    xgb = XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", early_stopping_rounds=25, n_jobs=-1,
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"xgb stopped at iteration {xgb.best_iteration} (val logloss {xgb.best_score:.4f})")
    p_xgb = xgb.predict_proba(Xte)[:, 1]
    results.append(evaluate("xgboost", yte, p_xgb))

    print()
    print(f"{'model':<22} {'log_loss':>10} {'brier':>10} {'auc':>8}")
    for r in results:
        print(f"{r['model']:<22} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {r['auc']:>8.4f}")

    imp = sorted(zip(feats, xgb.feature_importances_), key=lambda kv: -kv[1])
    print("\nxgb feature importance:")
    for n, v in imp:
        print(f"  {n:<14} {v:.3f}")

    cal = {}
    for label, p in [("xgb", p_xgb), ("logit", p_logit), ("distance", p_d)]:
        pt, pp = calibration_curve(yte, p, n_bins=20, strategy="quantile")
        cal[label] = {"pred": pp.tolist(), "true": pt.tolist()}

    xgb.save_model(os.path.join(args.out_dir, "xgboost.json"))

    test_out = test[[
        "season", "game_id", "team_id", "qtr", "distance", "angle", "is_three",
        "shot_type", "margin", "made", "score_value", "x_offense", "y_offense",
    ]].copy()
    test_out["xfg"] = p_xgb
    test_out.to_parquet(os.path.join(args.out_dir, "test_predictions.parquet"))

    report = {
        "metrics": results,
        "feature_importance": [{"feature": k, "importance": float(v)} for k, v in imp],
        "calibration": cal,
        "test_season": args.test_season,
        "n_train": len(train),
        "n_test": len(test),
        "base_rate": float(base),
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

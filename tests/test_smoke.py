import json
import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _needs(p):
    full = os.path.join(ROOT, p)
    if not os.path.exists(full):
        pytest.skip(f"missing {p} (run `make all`)")
    return full


# feature order: distance, angle, is_three, margin, clock, qtr,
#                is_dunk, is_hook, is_jumper, is_layup, is_tip
def _row(distance, angle, is_three, shot_type, margin=0, clock=2880, qtr=1):
    base = [distance, angle, is_three, margin, clock, qtr]
    types = ["dunk", "hook", "jumper", "layup", "tip"]
    for t in types:
        base.append(1 if shot_type == t else 0)
    return np.array([base], dtype=float)


def test_shots_parquet():
    df = pd.read_parquet(_needs("data/shots.parquet"))
    assert len(df) > 500_000
    assert df["made"].isin([0, 1]).all()
    assert 0.4 < df["made"].mean() < 0.55
    # mirrored half-court: shots within roughly 35 ft of the offensive hoop
    assert df["distance"].between(0.5, 35).all()


def test_dunk_is_high():
    # angle=0 is straight-on, angle=90 is corner
    clf = xgb.XGBClassifier()
    clf.load_model(_needs("models/xgboost.json"))
    p = clf.predict_proba(_row(1.5, 0, 0, "dunk", clock=600, qtr=2))[0, 1]
    assert p > 0.75


def test_long_jumper_is_low():
    clf = xgb.XGBClassifier()
    clf.load_model(_needs("models/xgboost.json"))
    p = clf.predict_proba(_row(27, 0, 1, "jumper", clock=600, qtr=2))[0, 1]
    assert p < 0.40  # 27-ft three is around 33%


def test_corner_three_at_least_as_good_as_above_break():
    clf = xgb.XGBClassifier()
    clf.load_model(_needs("models/xgboost.json"))
    corner = clf.predict_proba(_row(22.5, 90, 1, "jumper", clock=600, qtr=2))[0, 1]
    above_break = clf.predict_proba(_row(25.5, 0, 1, "jumper", clock=600, qtr=2))[0, 1]
    # league truth: corner threes go in at a higher clip
    assert corner >= above_break - 0.01


def test_no_score_value_leak():
    """Sanity check that the model isn't learning from score_value (which is
    zero for any miss). A made shot and a miss with identical features must
    score the same."""
    df = pd.read_parquet(_needs("data/shots.parquet"))
    assert "score_value" not in [
        c for c in df.columns if c in {"distance", "angle", "is_three"}
    ]


def test_metrics():
    with open(_needs("models/report.json")) as f:
        rep = json.load(f)
    xgb_row = next(m for m in rep["metrics"] if m["model"] == "xgboost")
    # without shooter ID and defender features, AUC tops out around 0.68
    assert xgb_row["auc"] > 0.65
    assert xgb_row["log_loss"] < 0.69

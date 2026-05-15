"""Filter pbp -> field-goal attempts with court coordinates.

We compute:
  - shot_distance (feet from the hoop nearest the offense)
  - shot_angle    (degrees from baseline, 90 = straight on)
  - is_three      (binary, derived from score_value or distance fallback)
  - shot_type     (jumper / drive / layup / dunk / hook / tip)
  - clock         (seconds remaining in the game)
  - margin        (score differential from shooting team's POV)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def shot_type_from_text(t: str) -> str:
    """Bucket the verbose type strings into something modelable."""
    if pd.isna(t):
        return "other"
    s = t.lower()
    if "dunk" in s:
        return "dunk"
    if "layup" in s or "finger roll" in s:
        return "layup"
    if "tip" in s:
        return "tip"
    if "hook" in s:
        return "hook"
    if "alley oop" in s:
        return "dunk"
    return "jumper"


def build(data_dir, out_path):
    files = sorted(glob.glob(os.path.join(data_dir, "pbp_*.parquet")))
    if not files:
        raise SystemExit("no pbp files")
    print(f"reading {len(files)} season files")

    cols = [
        "season", "game_id", "qtr", "period_number",
        "type_text", "shooting_play", "scoring_play", "score_value",
        "coordinate_x", "coordinate_y",
        "home_score", "away_score", "team_id", "home_team_id",
        "start_game_seconds_remaining",
    ]
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f, columns=cols)
        except Exception:
            df = pd.read_parquet(f)
            df = df[[c for c in cols if c in df.columns]]
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"  raw rows: {len(raw):,}")

    shots = raw[raw["shooting_play"] == True].copy()
    shots = shots[~shots["type_text"].str.contains("Free Throw", na=False)]
    shots = shots.dropna(subset=["coordinate_x", "coordinate_y"])
    print(f"  field-goal attempts with coords: {len(shots):,}")

    # the offense always shoots at the hoop that's farther down its half;
    # mirror everything to a single half-court so distance/angle make sense
    shots["x_offense"] = np.where(shots["coordinate_x"] >= 0,
                                  shots["coordinate_x"], -shots["coordinate_x"])
    shots["y_offense"] = np.where(shots["coordinate_x"] >= 0,
                                  shots["coordinate_y"], -shots["coordinate_y"])
    hoop_x, hoop_y = 41.75, 0.0
    dx = shots["x_offense"] - hoop_x
    dy = shots["y_offense"] - hoop_y
    shots["distance"] = np.sqrt(dx**2 + dy**2)
    # angle: 0 is straight from the corner (along baseline), 90 is straight on
    shots["angle"] = np.degrees(np.arctan2(np.abs(dy), -dx))
    # restrict to half-court attempts
    shots = shots[(shots["distance"] < 35) & (shots["distance"] > 0.5)]

    # made flag
    shots["made"] = (shots["scoring_play"] == True).astype(int)
    # three-pointer flag derived from court geometry, NOT score_value
    # (score_value is 0 for any miss, so using it would leak the outcome).
    # NBA arc: 23.75 ft above the break, 22 ft in the corners (|y| > 22).
    in_corner = shots["y_offense"].abs() > 22
    shots["is_three"] = (
        (in_corner & (shots["distance"] > 22)) |
        (~in_corner & (shots["distance"] > 23.75))
    ).astype(int)

    shots["shot_type"] = shots["type_text"].apply(shot_type_from_text)

    # game state
    margin_home = shots["home_score"] - shots["away_score"]
    shots["margin"] = np.where(shots["team_id"] == shots["home_team_id"],
                               margin_home, -margin_home)
    shots["clock"] = shots["start_game_seconds_remaining"].fillna(2880)

    out = shots[[
        "season", "game_id", "team_id", "qtr",
        "distance", "angle", "is_three", "shot_type",
        "margin", "clock", "made", "score_value", "type_text",
        "x_offense", "y_offense",
    ]].reset_index(drop=True)

    print(f"  final rows: {len(out):,}  fg%: {out['made'].mean():.3f}")
    print(f"  by type:")
    print(out.groupby("shot_type")["made"].agg(["size", "mean"]).round(3))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/shots.parquet")
    args = ap.parse_args()
    build(args.data, args.out)


if __name__ == "__main__":
    main()

"""Score a single shot.

  $ python scripts/predict.py --distance 23 --angle 0 --shot-type jumper \\
      --is-three --qtr 4 --clock 120 --margin -3
  xfg = 0.342

Coordinates default to the half-court center; pass --distance in feet
from the rim and --angle in degrees from the baseline (0 = straight on
under the rim, 90 = left wing).
"""

import argparse

import numpy as np
import xgboost as xgb

SHOT_TYPES = ["dunk", "hook", "jumper", "layup", "tip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, required=True,
                    help="distance from the basket in feet")
    ap.add_argument("--angle", type=float, default=0.0,
                    help="angle from the baseline in degrees")
    ap.add_argument("--shot-type", choices=SHOT_TYPES, required=True)
    ap.add_argument("--is-three", action="store_true",
                    help="3pt attempt (otherwise treated as 2pt)")
    ap.add_argument("--qtr", type=int, default=2)
    ap.add_argument("--clock", type=float, default=360.0,
                    help="seconds left in the quarter")
    ap.add_argument("--margin", type=int, default=0,
                    help="score margin from the shooter's team perspective")
    ap.add_argument("--model", default="models/xgboost.json")
    args = ap.parse_args()

    if args.distance < 0 or args.distance > 94:
        raise SystemExit("distance must be between 0 and 94 feet")
    if not 1 <= args.qtr <= 6:
        raise SystemExit("qtr must be 1-6 (5/6 is OT)")

    numeric = [args.distance, args.angle, 1 if args.is_three else 0,
               args.margin, args.clock, args.qtr]
    one_hot = [1 if args.shot_type == st else 0 for st in SHOT_TYPES]
    row = np.array([numeric + one_hot], dtype=float)

    clf = xgb.XGBClassifier()
    clf.load_model(args.model)
    p = clf.predict_proba(row)[0, 1]
    print(f"xfg = {p:.3f}")


if __name__ == "__main__":
    main()

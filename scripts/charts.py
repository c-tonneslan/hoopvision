"""Charts for the hoopvision writeup.

  - calibration.png      calibration curve across deciles
  - importance.png       xgb feature importance
  - court_xfg.png        league xFG% over the half-court
  - shot_volume.png      shot density heatmap
  - shotmaking.png       top "shotmakers" (actual FG% - xFG%, min 500 attempts)
"""

import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 10,
})


def draw_court(ax, color="#666"):
    """Sketch a half-court matching the dataset coords.

    Hoop at (41.75, 0), baseline at x=47, half-court line at x=0,
    sidelines at y=±25. Three-point corners at y=±22.
    """
    HOOP_X = 41.75
    BASELINE_X = 47
    # baseline + sidelines + half-court line
    ax.plot([BASELINE_X, BASELINE_X], [-25, 25], color=color, lw=1.2)
    ax.plot([0, BASELINE_X], [-25, -25], color=color, lw=1.2)
    ax.plot([0, BASELINE_X], [25, 25], color=color, lw=1.2)
    ax.plot([0, 0], [-25, 25], color=color, lw=1.2)
    # painted area (the key): 19 deep from baseline, 16 wide
    ax.add_patch(mpatches.Rectangle((BASELINE_X - 19, -8), 19, 16,
                                    fill=False, color=color, lw=1.2))
    # free-throw circle (at top of key, 15 ft from baseline)
    ax.add_patch(mpatches.Circle((BASELINE_X - 19, 0), 6,
                                 fill=False, color=color, lw=1.2))
    # backboard (just inside baseline, 4 ft from baseline)
    ax.plot([HOOP_X + 1.25, HOOP_X + 1.25], [-3, 3], color=color, lw=1.2)
    # hoop
    ax.add_patch(mpatches.Circle((HOOP_X, 0), 0.75, fill=False, color=color, lw=1.2))
    # restricted-area arc (4 ft from hoop, opening toward half-court)
    ax.add_patch(mpatches.Arc((HOOP_X, 0), 8, 8, theta1=90, theta2=270,
                              color=color, lw=1.2))
    # three-point arc (23.75 ft from hoop, corner break at y=±22)
    corner_x = HOOP_X - np.sqrt(23.75**2 - 22**2)
    # angle from hoop center to (corner_x, 22) is atan2(22, corner_x - HOOP_X)
    theta_top = np.degrees(np.arctan2(22, corner_x - HOOP_X))      # ~112 deg
    theta_bot = np.degrees(np.arctan2(-22, corner_x - HOOP_X))     # ~-112 deg
    ax.add_patch(mpatches.Arc((HOOP_X, 0), 47.5, 47.5,
                              theta1=theta_top, theta2=theta_bot + 360,
                              color=color, lw=1.2))
    # corner three lines
    ax.plot([BASELINE_X, corner_x], [22, 22], color=color, lw=1.2)
    ax.plot([BASELINE_X, corner_x], [-22, -22], color=color, lw=1.2)
    ax.set_xlim(-1, 48)
    ax.set_ylim(-26, 26)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def calibration_chart(report, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    for k, color, label in [
        ("xgb", "#0a84ff", "xgboost"),
        ("logit", "#ff8c00", "logistic"),
        ("distance", "#888", "distance only"),
    ]:
        c = report["calibration"][k]
        ax.plot(c["pred"], c["true"], marker="o", ms=4, color=color, label=label)
    ax.set_xlabel("predicted P(make)")
    ax.set_ylabel("empirical make rate")
    ax.set_title("calibration on 2023-24 NBA shots")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def importance_chart(report, out_path):
    fi = report["feature_importance"]
    names = [f["feature"] for f in fi][::-1]
    vals = [f["importance"] for f in fi][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color="#0a84ff", alpha=0.85)
    ax.set_xlabel("importance")
    ax.set_title("xgboost feature importance")
    ax.grid(alpha=0.25, axis="x")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def court_xfg(out_path):
    df = pd.read_parquet("models/test_predictions.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_court(ax)
    # 2d binned average xFG%
    H_sum, xe, ye = np.histogram2d(
        df["x_offense"], df["y_offense"], bins=[24, 26],
        range=[[24, 48], [-26, 26]], weights=df["xfg"]
    )
    H_cnt, _, _ = np.histogram2d(
        df["x_offense"], df["y_offense"], bins=[24, 26],
        range=[[24, 48], [-26, 26]]
    )
    with np.errstate(invalid="ignore"):
        H_avg = np.where(H_cnt > 30, H_sum / H_cnt, np.nan)
    im = ax.imshow(H_avg.T, origin="lower", extent=[24, 48, -26, 26],
                   cmap="viridis", vmin=0.2, vmax=0.75, alpha=0.85)
    draw_court(ax)
    fig.colorbar(im, ax=ax, label="xFG%")
    ax.set_title("model xFG% by court location (2023-24 NBA)")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def shot_volume(out_path):
    df = pd.read_parquet("models/test_predictions.parquet")
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_court(ax)
    H, xe, ye = np.histogram2d(
        df["x_offense"], df["y_offense"], bins=[40, 40],
        range=[[24, 48], [-26, 26]]
    )
    H = np.log1p(H)
    im = ax.imshow(H.T, origin="lower", extent=[24, 48, -26, 26],
                   cmap="hot", alpha=0.85)
    draw_court(ax)
    fig.colorbar(im, ax=ax, label="log(1 + attempts)")
    ax.set_title("shot density by court location (2023-24 NBA)")
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def shotmaking(out_path, min_shots=500):
    df = pd.read_parquet("models/test_predictions.parquet")
    # we don't have player names in the pbp shot rows directly; use team for now
    # but the actual per-shooter name is in athlete_id_1 -> needs lookup.
    # we'll group by team_id and grab top "team shot-quality" overperformers.
    # team scoreboard: which teams beat xFG% by the most
    g = df.groupby("team_id").agg(
        attempts=("made", "size"),
        fg_pct=("made", "mean"),
        xfg=("xfg", "mean"),
    )
    g["shotmaking"] = g["fg_pct"] - g["xfg"]
    g = g[g["attempts"] >= min_shots].sort_values("shotmaking", ascending=False)
    top = g.head(10).reset_index()
    bot = g.tail(10).reset_index().iloc[::-1]

    def render(d, title, color, out):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.axis("off")
        ax.set_title(title, loc="left", fontweight="bold")
        rows = [[int(r["team_id"]), f"{r['attempts']:.0f}", f"{r['fg_pct']:.3f}",
                 f"{r['xfg']:.3f}", f"{r['shotmaking']:+.3f}"] for _, r in d.iterrows()]
        t = ax.table(cellText=rows,
                     colLabels=["team_id", "attempts", "fg%", "xfg%", "fg - xfg"],
                     cellLoc="left", colLoc="left", loc="upper left")
        t.auto_set_font_size(False)
        t.set_fontsize(10)
        t.scale(1, 1.4)
        for j in range(5):
            c = t[(0, j)]
            c.set_facecolor(color)
            c.set_text_props(color="white", fontweight="bold")
        fig.tight_layout()
        fig.savefig(out)
        print(f"  wrote {out}")

    render(top, "Best shotmaking teams (FG% - xFG%, 2023-24)", "#34c759",
           out_path.replace(".png", "_top.png"))
    render(bot, "Worst shotmaking teams (FG% - xFG%, 2023-24)", "#ff3b30",
           out_path.replace(".png", "_bot.png"))


def main():
    os.makedirs("charts", exist_ok=True)
    with open("models/report.json") as f:
        report = json.load(f)
    calibration_chart(report, "charts/calibration.png")
    importance_chart(report, "charts/importance.png")
    court_xfg("charts/court_xfg.png")
    shot_volume("charts/shot_volume.png")
    shotmaking("charts/shotmaking.png")


if __name__ == "__main__":
    main()

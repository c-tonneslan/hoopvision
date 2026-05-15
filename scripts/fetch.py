"""Pull NBA play-by-play parquet files from the hoopR-nba-data mirror."""

import argparse
import os
import urllib.request

URL = (
    "https://github.com/sportsdataverse/hoopR-nba-data/raw/main/"
    "nba/pbp/parquet/play_by_play_{year}.parquet"
)


def fetch(years, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for y in years:
        path = os.path.join(out_dir, f"pbp_{y}.parquet")
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print(f"  have {y}")
            continue
        try:
            urllib.request.urlretrieve(URL.format(year=y), path)
            sz = os.path.getsize(path) / 1e6
            print(f"  wrote {y} ({sz:.1f} MB)")
        except Exception as e:
            print(f"  skip {y}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2022)
    ap.add_argument("--end", type=int, default=2024)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    fetch(range(args.start, args.end + 1), args.out)


if __name__ == "__main__":
    main()

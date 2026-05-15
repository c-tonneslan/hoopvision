.PHONY: all fetch dataset train charts test clean

all: fetch dataset train charts

fetch:
	python3 scripts/fetch.py --start 2022 --end 2024

dataset:
	python3 scripts/build_dataset.py

train:
	python3 scripts/train.py

charts:
	python3 scripts/charts.py

test:
	python3 -m pytest tests/ -v

clean:
	rm -f data/*.parquet models/* charts/*.png

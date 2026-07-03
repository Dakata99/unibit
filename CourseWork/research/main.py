#!/usr/bin/env python3

import glob
import argparse
import kagglehub
import pandas as pd
from pathlib import Path

TOPICS = [
    # {
    #     "website": "",
    #     "dataset": "",
    #     "file": ""
    # }
    {
        # NO!
        "website": "kaggle",
        "dataset": "mdrakiburrahman10/hepatitis-c-virus-hcv-for-egyptian-patients",
        "file": "Discretization-Criteria.csv"
    },
    {   # POSSIBLE!
        "website": "kaggle",
        "dataset": "paramjeetsinghds/indian-liver-disease-dataset",
        "file": "Training_indian_liver_disease_dataset.csv"
    },
    {
        "website": "kaggle",
        "dataset": "utkarshx27/non-alcohol-fatty-liver-disease",
        "file": None,
    }
]

def download(dataset: str):
    # Download latest version
    path = kagglehub.dataset_download(dataset)
    print("Dataset downloaded to:", path)
    return path

def info(path: Path, file: Path):
    # Load the CSV (adjust the filename if needed)
    if file is None:
        csv_files = glob.glob(str(Path(path) / "*.csv"))
    else:
        csv_files = [ Path(path) / file ]

    for csv_file in csv_files:
        print(f'------------------------- {Path(csv_file).name} -------------------------')
        df = pd.read_csv(Path(path) / csv_file)
        print('------------------------- HEAD -------------------------')
        print(df.head())
        print('------------------------- INFO -------------------------')
        print(df.info())
        print('------------------------- FEATURES -------------------------')
        print(df.columns)
        print('------------------------- DATASET -------------------------')
        print(df)


def main():
    parser = argparse.ArgumentParser(prog="research")
    parser.add_argument("--index", type=int, help="Dataset index to check")
    parser.add_argument("--print", action='store_true', help="Print list of datasets")
    args = parser.parse_args()

    if args.print:
        for topic in TOPICS:
            idx = 0
            print(
                f"Index: {idx}\n"
                f"\t{TOPICS[idx]}"
            )
            idx += 1
    elif args.index is not None:
        assert args.index > -1 and args.index < len(TOPICS), "Not a valid index"
        topic = TOPICS[args.index]
        path = download(topic['dataset'])
        info(path, topic['file'])

if __name__ == "__main__":
    main()
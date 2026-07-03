#!/usr/bin/env python3

"""
Docstring for research.pp
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


# ----------------------------
# URLs (NHANES 2017–2018, suffix _J)
# ----------------------------
URLS = {
    "DEMO_J.xpt":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DEMO_J.xpt",
    "BIOPRO_J.xpt":"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/BIOPRO_J.xpt",
    "TCHOL_J.xpt": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/TCHOL_J.xpt",
    "HEPC_J.xpt":  "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEPC_J.xpt",
    "HEQ_J.xpt":   "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/HEQ_J.xpt",
}


# ----------------------------
# UCI-like feature mapping
# UCI: Age, Sex, ALB, ALP, ALT, AST, BIL, CHOL, CREA, GGT, PROT
# NHANES equivalents:
#   Age: RIDAGEYR
#   Sex: RIAGENDR
#   ALB: LBXSAL
#   ALP: LBXSAPSI
#   ALT: LBXSATSI
#   AST: LBXSASSI
#   BIL: LBXSTB
#   CHOL: LBXTC  (from TCHOL_J, preferred)
#   CREA: LBXSCR
#   GGT: LBXSGTSI
#   PROT: LBXSTP
# ----------------------------
NHANES_TO_UCI = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "LBXSAL": "ALB",
    "LBXSAPSI": "ALP",
    "LBXSATSI": "ALT",
    "LBXSASSI": "AST",
    "LBXSTB": "BIL",
    "LBXTC": "CHOL",
    "LBXSCR": "CREA",
    "LBXSGTSI": "GGT",
    "LBXSTP": "PROT",
}

UCI_COL_ORDER = ["SEQN", "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHOL", "CREA", "GGT", "PROT", "y"]


def download_if_missing(data_dir: Path, filename: str) -> None:
    path = data_dir / filename
    if path.exists():
        return
    if requests is None:
        raise RuntimeError("requests is not installed. Install it or download XPT files manually.")
    url = URLS[filename]
    print(f"⬇️  Downloading {filename} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)


def read_xpt(path: Path) -> pd.DataFrame:
    df = pd.read_sas(path, format="xport")
    df.columns = [c.decode("utf-8") if isinstance(c, (bytes, bytearray)) else str(c) for c in df.columns]
    if "SEQN" not in df.columns:
        raise ValueError(f"{path.name}: missing SEQN")
    df["SEQN"] = pd.to_numeric(df["SEQN"], errors="coerce").astype("Int64")
    df = df[df["SEQN"].notna()].copy()
    return df


def assert_unique_seqn(df: pd.DataFrame, name: str) -> None:
    if df["SEQN"].duplicated().any():
        dups = df.loc[df["SEQN"].duplicated(keep=False), "SEQN"].dropna().astype("int64").unique()[:20]
        raise ValueError(f"{name}: SEQN not unique (merge would multiply rows). Example dups: {dups.tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("./nhanes_2017_2018"), help="Folder for XPT files")
    ap.add_argument("--label", choices=["rna", "ab_confirmed", "self_report"], default="rna",
                    help="Which HCV label to build (default: rna)")
    ap.add_argument("--sex-as", choices=["numeric", "mf", "MaleFemale"], default="mf",
                    help="Encode Sex as 1/2, m/f, or Male/Female")
    ap.add_argument("--drop-missing-any-feature", action="store_true",
                    help="Drop rows missing ANY UCI-like feature (recommended for clean ML table)")
    ap.add_argument("--out", type=Path, default=Path("nhanes_uci_like_with_label.csv"))
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Decide which files are needed
    needed_files = ["DEMO_J.xpt", "BIOPRO_J.xpt", "TCHOL_J.xpt", "HEPC_J.xpt"]
    if args.label == "self_report":
        needed_files.append("HEQ_J.xpt")

    # Download if missing
    for fn in needed_files:
        download_if_missing(data_dir, fn)

    # Load
    demo = read_xpt(data_dir / "DEMO_J.xpt")
    biopro = read_xpt(data_dir / "BIOPRO_J.xpt")
    tchol = read_xpt(data_dir / "TCHOL_J.xpt")
    hepc = read_xpt(data_dir / "HEPC_J.xpt")
    heq = None
    if args.label == "self_report":
        heq = read_xpt(data_dir / "HEQ_J.xpt")

    # Keep minimal columns only (safer + cleaner)
    demo_keep = ["SEQN", "RIDSTATR", "RIDAGEYR", "RIAGENDR"]
    biopro_keep = ["SEQN", "LBXSAL", "LBXSAPSI", "LBXSATSI", "LBXSASSI", "LBXSTB", "LBXSCR", "LBXSGTSI", "LBXSTP"]
    tchol_keep = ["SEQN", "LBXTC"]
    hepc_keep = ["SEQN", "LBXHCR", "LBDHCI"]  # label sources

    demo = demo[[c for c in demo_keep if c in demo.columns]].copy()
    biopro = biopro[[c for c in biopro_keep if c in biopro.columns]].copy()
    tchol = tchol[[c for c in tchol_keep if c in tchol.columns]].copy()
    hepc = hepc[[c for c in hepc_keep if c in hepc.columns]].copy()

    if heq is not None:
        heq = heq[[c for c in ["SEQN", "HEQ030"] if c in heq.columns]].copy()

    # Sanity checks (avoid row multiplication)
    for name, df in [("DEMO", demo), ("BIOPRO", biopro), ("TCHOL", tchol), ("HEPC", hepc)]:
        assert_unique_seqn(df, name)
    if heq is not None:
        assert_unique_seqn(heq, "HEQ")

    # Keep only examined participants (RIDSTATR == 2) because labs require MEC exam
    if "RIDSTATR" in demo.columns:
        demo = demo[demo["RIDSTATR"] == 2].copy()

    # Merge (DEMO base; one_to_one validation)
    df = demo.merge(biopro, on="SEQN", how="left", validate="one_to_one")
    df = df.merge(tchol, on="SEQN", how="left", validate="one_to_one")
    df = df.merge(hepc, on="SEQN", how="left", validate="one_to_one")
    if heq is not None:
        df = df.merge(heq, on="SEQN", how="left", validate="one_to_one")

    # Build label y
    if args.label == "rna":
        # LBXHCR: 1 Positive, 2 Negative, 3 Negative Screening HCV Antibody
        # y=1 for 1; y=0 for 2 or 3; missing otherwise
        df["y"] = df["LBXHCR"].map({1: 1, 2: 0, 3: 0})
    elif args.label == "ab_confirmed":
        # LBDHCI: 1 Positive, 2 Negative, 3 Negative Screening HCV Antibody, 4 Positive HCV RNA
        # treat 1 or 4 as positive; 2 or 3 as negative
        df["y"] = df["LBDHCI"].map({1: 1, 4: 1, 2: 0, 3: 0})
    else:  # self_report
        # HEQ030: 1 Yes, 2 No (7/9 -> missing)
        df["y"] = df["HEQ030"].map({1: 1, 2: 0})

    # Keep only UCI-like features + y (rename)
    keep_nhanes = ["SEQN"] + list(NHANES_TO_UCI.keys()) + ["y"]
    keep_nhanes = [c for c in keep_nhanes if c in df.columns]
    out_df = df[keep_nhanes].rename(columns=NHANES_TO_UCI).copy()

    # Encode sex
    if "Sex" in out_df.columns:
        if args.sex_as == "mf":
            out_df["Sex"] = out_df["Sex"].map({1: "m", 2: "f"})
        elif args.sex_as == "MaleFemale":
            out_df["Sex"] = out_df["Sex"].map({1: "Male", 2: "Female"})
        # numeric: leave as 1/2

    # Optionally drop rows missing any feature (and/or missing label)
    feature_cols = [c for c in UCI_COL_ORDER if c not in ("SEQN", "y") and c in out_df.columns]
    if args.drop_missing_any_feature:
        before = len(out_df)
        out_df = out_df.dropna(subset=feature_cols + ["y"]).copy()
        after = len(out_df)
        print(f"🧹 Dropped rows with any missing feature/label: {before} -> {after}")

    # Order columns
    final_cols = [c for c in UCI_COL_ORDER if c in out_df.columns]
    out_df = out_df[final_cols].copy()

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"✅ Saved: {args.out.resolve()}")
    print("Columns:", list(out_df.columns))
    print("Label distribution (y):")
    print(out_df["y"].value_counts(dropna=False))

    print('--------------------------------------')

    # Path to the CSV you exported (change if needed)
    csv_path = Path(args.out)

    # Load as a DataFrame
    df = pd.read_csv(csv_path)

    print(df.shape)
    print(df.columns.tolist())
    print((df['y'] == 1).sum())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

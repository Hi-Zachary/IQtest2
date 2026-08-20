"""Prepare AGIQA-3K / AIGCIQA2023 annotation files for MSQRNet.

Converts the official annotation (name,prompt,...,mos_quality,...,mos_align)
into the `mos_joint.xlsx` that the MSQRNet AGIQA3k dataset expects
(columns: name, prompt, mos_quality, mos_align).

AGIQA-3K:
    python prepare_data.py --csv data/aigc_qa_3k/data.csv \
        --out data/aigc_qa_3k/mos_joint.xlsx

AIGCIQA2023 (already provided as mos_joint_aigciqa2023.xlsx):
    python prepare_data.py --csv <AIGCIQA2023 annotation csv> \
        --out data/aigc_qa_2023/mos_joint_aigciqa2023.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/aigc_qa_3k/data.csv",
        help="official annotation csv",
    )
    parser.add_argument(
        "--out",
        default="data/aigc_qa_3k/mos_joint.xlsx",
        help="output mos_joint.xlsx for the dataset",
    )
    args = parser.parse_args()

    src = Path(args.csv)
    out = Path(args.out)
    assert src.exists(), f"data.csv not found: {src}"

    df = pd.read_csv(src)
    df2 = df[["name", "prompt", "mos_quality", "mos_align"]].copy()
    df2.columns = ["name", "prompt", "mos_quality", "mos_align"]

    out.parent.mkdir(parents=True, exist_ok=True)
    df2.to_excel(out, index=False)
    print(f"saved: {out}  rows={len(df2)}")


if __name__ == "__main__":
    main()

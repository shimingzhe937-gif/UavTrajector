#!/usr/bin/env python3
"""Aggregate fair model-complexity results from comparison_seed_results.csv.

This script does not retrain models. It summarizes the hardware measurements
already produced by the formal comparison script and derives throughput. K=1
and K=100 latency are deliberately kept in separate columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def latex_table(frame: pd.DataFrame) -> str:
    """Render a booktabs table without pandas' optional Jinja2 backend."""
    escapes = str.maketrans({
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"\(\pm\)",
    })
    column_format = "l" + "c" * (len(frame.columns) - 1)
    lines = [rf"\begin{{tabular}}{{{column_format}}}", r"\toprule"]
    lines.append(" & ".join(str(name).translate(escapes) for name in frame.columns) + r" \\")
    lines.append(r"\midrule")
    for row in frame.itertuples(index=False, name=None):
        lines.append(" & ".join(str(value).translate(escapes) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="comparison_seed_results.csv")
    ap.add_argument("--out", default=None, help="output directory; default: input parent/complexity")
    args = ap.parse_args()

    src = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve() if args.out else src.parent / "complexity"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    if "RunStatus" in df:
        df = df[df["RunStatus"].astype(str).str.lower().eq("completed")].copy()
    if "Error" in df:
        df = df[df["Error"].isna() | df["Error"].astype(str).isin(["", "nan", "None"])].copy()

    required = ["Model", "Seed", "Params_M", "FLOPs_G", "Latency_K1_ms"]
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    metrics = [c for c in ["Params_M", "FLOPs_G", "Latency_K1_ms", "Latency_K100_ms"] if c in df]
    for c in metrics:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Throughput_K1_traj_s"] = 1000.0 / df["Latency_K1_ms"].replace(0, np.nan)
    if "Latency_K100_ms" in df:
        df["Throughput_K100_sets_s"] = 1000.0 / df["Latency_K100_ms"].replace(0, np.nan)
        metrics += ["Throughput_K1_traj_s", "Throughput_K100_sets_s"]
    else:
        metrics += ["Throughput_K1_traj_s"]

    # Params/FLOPs are structural; latency varies by run and is summarized over seeds.
    rows = []
    for model, g in df.groupby("Model", sort=False):
        row = {"Model": model, "N": int(g["Seed"].nunique())}
        for c in metrics:
            x = pd.to_numeric(g[c], errors="coerce").dropna()
            row[f"{c}_mean"] = float(x.mean()) if len(x) else np.nan
            row[f"{c}_std"] = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "complexity_mean_std.csv", index=False, encoding="utf-8-sig")

    paper = pd.DataFrame({"Model": summary["Model"]})
    labels = {
        "Params_M": "Params (M)", "FLOPs_G": "FLOPs (G)",
        "Latency_K1_ms": "Latency K=1 (ms)", "Latency_K100_ms": "Latency K=100 (ms)",
        "Throughput_K1_traj_s": "Throughput K=1 (traj/s)",
        "Throughput_K100_sets_s": "Throughput K=100 (sets/s)",
    }
    for c in metrics:
        paper[labels[c]] = [fmt(m, s) for m, s in zip(summary[f"{c}_mean"], summary[f"{c}_std"])]
    paper.to_csv(out / "complexity_paper_table.csv", index=False, encoding="utf-8-sig")
    (out / "complexity_paper_table.tex").write_text(
        latex_table(paper),
        encoding="utf-8",
    )
    print(paper.to_string(index=False))
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()

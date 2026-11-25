#!/usr/bin/env python3
"""Create fair, paper-ready multi-seed summaries for HMM UAV experiments.

Independent replication unit: random seed. Point ADE/FDE are reconstructed by
sample-count weighting of the three disjoint maneuver groups. Best-of-K
minADE/minFDE are retained separately and never relabelled as point metrics.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


STATE_NAME = {"low": "Low", "mid": "Moderate", "moderate": "Moderate", "high": "High"}
MAIN_METRICS = ["ADE", "FDE", "ES", "DeltaNLL", "PIC95", "MIW95"]
STATE_METRICS = MAIN_METRICS + [
    "Ev", "Ea", "Ej", "A_KEER", "NJER", "KPC",
    "ViolationRate", "CandidateViolationRate",
]
STEP_SOURCE = {
    "dist_error_m": "DistanceError",
    "step_RMSE": "RMSE",
    "dist_acc_pct": "DistanceAccuracy",
    "dir_acc_pct": "DirectionAccuracy",
    "DeltaNLL": "DeltaNLL",
    "mPIC95": "PIC95",
    "MIW95": "MIW95",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--model-name", default="Ours")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def seed_id(path):
    match = re.fullmatch(r"seed_(\d+)", path.name)
    if not match:
        raise ValueError(f"Invalid seed directory: {path}")
    return int(match.group(1))


def number(value):
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def summarise(frame, identifiers, metrics):
    rows = []
    for keys, part in frame.groupby(identifiers, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(identifiers, keys))
        for metric in metrics:
            if metric not in part:
                continue
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                **identity,
                "Metric": metric,
                "NSeeds": int(values.size),
                "Mean": float(values.mean()),
                "SD": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                "Min": float(values.min()),
                "Max": float(values.max()),
            })
    return pd.DataFrame(rows)


def formatted(summary):
    output = summary.copy()
    def render(row):
        decimals = 2 if row["Metric"] in {"PIC95", "DistanceAccuracy", "DirectionAccuracy"} else 4
        if pd.isna(row["SD"]):
            return f"{row['Mean']:.{decimals}f}"
        return f"{row['Mean']:.{decimals}f} ± {row['SD']:.{decimals}f}"
    output["MeanSD"] = output.apply(render, axis=1)
    return output


def latex_value(summary, metric):
    row = summary[summary["Metric"] == metric]
    if row.empty:
        return "--"
    row = row.iloc[0]
    decimals = 2 if metric == "PIC95" else 4
    return f"{row['Mean']:.{decimals}f} $\\pm$ {row['SD']:.{decimals}f}"


def escape_latex(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_global_latex(summary, model, path, n_seeds):
    cells = [latex_value(summary, metric) for metric in MAIN_METRICS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Overall results across {n_seeds} independent random seeds. Values are mean $\pm$ standard deviation.}}",
        r"\label{tab:hmm_multiseed_global}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & ADE$\downarrow$ & FDE$\downarrow$ & ES$\downarrow$ & $\Delta$NLL$\downarrow$ & PIC$_{95}$ (\%) & MIW$_{95}\downarrow$ \\",
        r"\midrule",
        escape_latex(model) + " & " + " & ".join(cells) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_state_latex(summary, model, path, n_seeds):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{State-wise results across {n_seeds} independent random seeds. Values are mean $\pm$ standard deviation.}}",
        r"\label{tab:hmm_multiseed_states}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Method & State & ADE$\downarrow$ & FDE$\downarrow$ & ES$\downarrow$ & $\Delta$NLL$\downarrow$ & PIC$_{95}$ (\%) & MIW$_{95}\downarrow$ \\",
        r"\midrule",
    ]
    for state in ["Low", "Moderate", "High"]:
        part = summary[summary["State"] == state]
        cells = [latex_value(part, metric) for metric in MAIN_METRICS]
        lines.append(escape_latex(model) + " & " + state + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = arguments()
    root = Path(args.result_dir).expanduser().resolve()
    output = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "paper_summary_v2"
    output.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        [path for path in root.glob("seed_*") if path.is_dir()], key=seed_id
    )
    complete, incomplete = [], []
    for directory in candidates:
        missing = [name for name in ["test_probabilistic_metrics.csv", "test_maneuver_detail.csv"] if not (directory / name).exists()]
        if missing:
            incomplete.append({"Seed": seed_id(directory), "Missing": ";".join(missing)})
        else:
            complete.append(directory)
    if incomplete:
        pd.DataFrame(incomplete).to_csv(output / "incomplete_seeds.csv", index=False)
    if args.expected_seeds and len(complete) != args.expected_seeds:
        raise RuntimeError(f"Expected {args.expected_seeds} complete seeds, found {len(complete)}")
    if not complete:
        raise RuntimeError(f"No complete seed outputs found in {root}")

    global_rows, state_rows, step_rows = [], [], []
    for directory in complete:
        seed = seed_id(directory)
        probability = pd.read_csv(directory / "test_probabilistic_metrics.csv").iloc[0]
        detail = pd.read_csv(directory / "test_maneuver_detail.csv")
        detail["group_key"] = detail["group_key"].astype(str).str.lower()
        group_rows = detail.drop_duplicates("group_key").copy()
        counts = pd.to_numeric(group_rows["n_samples"], errors="coerce")
        total = counts.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"Invalid n_samples in {directory}")

        global_rows.append({
            "Model": args.model_name,
            "Seed": seed,
            "ADE": float((pd.to_numeric(group_rows["overall_ADE"], errors="coerce") * counts).sum() / total),
            "FDE": float((pd.to_numeric(group_rows["overall_FDE"], errors="coerce") * counts).sum() / total),
            "minADE": number(probability.get("minADE", np.nan)),
            "minFDE": number(probability.get("minFDE", np.nan)),
            "ES": number(probability.get("EnergyScore", np.nan)),
            "DeltaNLL": number(probability.get("DeltaNLL", np.nan)),
            "PIC95": number(probability.get("mPIC95", np.nan)),
            "MIW95": number(probability.get("MIW95", np.nan)),
            "StateAcc": number(probability.get("Intent_Acc", np.nan)),
            "StateF1": number(probability.get("Intent_F1", np.nan)),
        })

        for group_key, part in detail.groupby("group_key", sort=False):
            first = part.iloc[0]
            state_row = {
                "Model": args.model_name,
                "Seed": seed,
                "State": STATE_NAME.get(group_key, group_key),
                "NTest": int(number(first["n_samples"])),
                "ADE": number(first["overall_ADE"]),
                "FDE": number(first["overall_FDE"]),
                "ES": number(first["overall_ES"]),
                "DeltaNLL": float(pd.to_numeric(part["DeltaNLL"], errors="coerce").mean()),
                "PIC95": number(first["overall_mPIC95"]),
                "MIW95": number(first["overall_MIW95"]),
            }
            for metric in ["Ev", "Ea", "Ej", "A_KEER", "NJER", "KPC", "ViolationRate", "CandidateViolationRate"]:
                state_row[metric] = number(first.get(metric, np.nan))
            state_rows.append(state_row)

            for _, row in part.iterrows():
                base = {
                    "Model": args.model_name,
                    "Seed": seed,
                    "State": STATE_NAME.get(group_key, group_key),
                    "Step": int(number(row["step"])),
                }
                for source, metric in STEP_SOURCE.items():
                    if source in row:
                        step_rows.append({**base, "Metric": metric, "Value": number(row[source])})

    global_seed = pd.DataFrame(global_rows)
    state_seed = pd.DataFrame(state_rows)
    step_seed = pd.DataFrame(step_rows)
    global_seed.to_csv(output / "global_seed_results.csv", index=False)
    state_seed.to_csv(output / "state_seed_results.csv", index=False)

    global_summary = summarise(
        global_seed, ["Model"],
        MAIN_METRICS + ["minADE", "minFDE", "StateAcc", "StateF1"],
    )
    state_summary = summarise(state_seed, ["Model", "State"], STATE_METRICS)
    step_summary = summarise(
        step_seed.pivot_table(
            index=["Model", "Seed", "State", "Step"],
            columns="Metric", values="Value", aggfunc="mean"
        ).reset_index(),
        ["Model", "State", "Step"],
        list(STEP_SOURCE.values()),
    )
    global_summary.to_csv(output / "global_mean_std_long.csv", index=False)
    state_summary.to_csv(output / "state_mean_std_long.csv", index=False)
    step_summary.to_csv(output / "stepwise_mean_std_long.csv", index=False)
    formatted(global_summary).to_csv(output / "paper_global_formatted.csv", index=False)
    formatted(state_summary).to_csv(output / "paper_states_formatted.csv", index=False)
    formatted(step_summary).to_csv(output / "paper_stepwise_formatted.csv", index=False)

    write_global_latex(global_summary, args.model_name, output / "paper_table_global.tex", len(complete))
    write_state_latex(state_summary, args.model_name, output / "paper_table_states.tex", len(complete))
    pd.DataFrame([{
        "Model": args.model_name,
        "NSeeds": len(complete),
        "Seeds": " ".join(str(seed_id(path)) for path in complete),
        "ReplicationUnit": "independent random seed",
        "Summary": "mean +/- sample SD (ddof=1)",
        "PointMetricNote": "ADE/FDE are sample-count-weighted point metrics; minADE/minFDE are reported separately",
    }]).to_csv(output / "summary_protocol.csv", index=False)
    print(f"Summarised {len(complete)} seeds with fair point metrics: {output}")


if __name__ == "__main__":
    main()

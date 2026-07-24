from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare AdamW grokking speed with one or more "
            "Muon unembedding-learning-rate runs."
        )
    )

    parser.add_argument(
        "--adamw",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--muon",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/grokking_speed_comparison.html"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/grokking_speed_comparison.csv"
        ),
    )

    return parser.parse_args()


def normalize_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    return frame.rename(
        columns={
            "train_acc": "train_accuracy",
            "test_acc": "test_accuracy",
        }
    ).sort_values(
        "step"
    ).drop_duplicates(
        "step"
    )


def first_sustained_step(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    values = frame[column].to_numpy()
    steps = frame["step"].to_numpy()

    for index in range(
        len(values) - consecutive + 1
    ):
        if np.all(
            values[
                index:index + consecutive
            ] >= threshold
        ):
            return int(
                steps[index]
            )

    return None


def first_threshold_step(
    frame: pd.DataFrame,
    column: str,
    threshold: float,
) -> int | None:
    matching = frame.loc[
        frame[column] >= threshold,
        "step",
    ]

    if matching.empty:
        return None

    return int(
        matching.iloc[0]
    )


def summarize_run(
    label: str,
    frame: pd.DataFrame,
    adamw_grokking_step: int | None,
) -> dict:
    memorization_step = first_sustained_step(
        frame,
        "train_accuracy",
        0.999,
    )

    grokking_step = first_sustained_step(
        frame,
        "test_accuracy",
        0.95,
    )

    collapse_count = (
        int(frame["collapse_detected"].sum())
        if "collapse_detected" in frame.columns
        else 0
    )

    speedup = None

    if (
        adamw_grokking_step is not None
        and grokking_step is not None
    ):
        speedup = (
            adamw_grokking_step
            / grokking_step
        )

    return {
        "run": label,
        "sustained_99.9_train_step": (
            memorization_step
        ),
        "sustained_95_test_step": (
            grokking_step
        ),
        "memorization_plateau_steps": (
            None
            if (
                memorization_step is None
                or grokking_step is None
            )
            else (
                grokking_step
                - memorization_step
            )
        ),
        "first_99_test_step": (
            first_threshold_step(
                frame,
                "test_accuracy",
                0.99,
            )
        ),
        "first_100_test_step": (
            first_threshold_step(
                frame,
                "test_accuracy",
                0.999999,
            )
        ),
        "collapse_evaluations": (
            collapse_count
        ),
        "maximum_test_accuracy": float(
            frame["test_accuracy"].max()
        ),
        "final_test_accuracy": float(
            frame.iloc[-1]["test_accuracy"]
        ),
        "stable_by_definition": bool(
            collapse_count == 0
            and frame.iloc[-1]["test_accuracy"] >= 0.99
        ),
        "speedup_vs_adamw": speedup,
    }


def main() -> None:
    args = parse_arguments()

    if not args.adamw.exists():
        raise FileNotFoundError(
            f"Missing AdamW CSV: {args.adamw}"
        )

    adamw = normalize_columns(
        pd.read_csv(args.adamw)
    )

    adamw_grokking_step = (
        first_sustained_step(
            adamw,
            "test_accuracy",
            0.95,
        )
    )

    runs: list[tuple[str, pd.DataFrame]] = [
        ("AdamW", adamw)
    ]

    for path in args.muon:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Muon CSV: {path}"
            )

        frame = normalize_columns(
            pd.read_csv(path)
        )

        if (
            "unembedding_learning_rate"
            in frame.columns
        ):
            learning_rate = float(
                frame[
                    "unembedding_learning_rate"
                ].iloc[0]
            )

            label = (
                f"Muon unembedding lr="
                f"{learning_rate:g}"
            )
        else:
            label = path.stem

        runs.append(
            (label, frame)
        )

    summary_rows = [
        summarize_run(
            label,
            frame,
            adamw_grokking_step,
        )
        for label, frame in runs
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        args.summary_output,
        index=False,
    )

    figure = go.Figure()

    for label, frame in runs:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["test_accuracy"],
                mode="lines",
                name=label,
            )
        )

    figure.update_layout(
        title=(
            "Grokking speed: AdamW versus "
            "stable Muon regimes"
        ),
        xaxis_title="Training step",
        yaxis_title="Test accuracy",
        yaxis={
            "range": [-0.02, 1.02],
        },
        template="plotly_white",
        hovermode="x unified",
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(args.output),
        include_plotlyjs="cdn",
    )

    print(f"Saved plot to: {args.output}")
    print(
        f"Saved summary to: "
        f"{args.summary_output}"
    )
    print()
    print(
        summary.to_string(
            index=False
        )
    )

    stable_muon = summary.loc[
        (
            summary["run"] != "AdamW"
        )
        & (
            summary["stable_by_definition"]
        )
        & (
            summary[
                "sustained_95_test_step"
            ].notna()
        )
    ].sort_values(
        "sustained_95_test_step"
    )

    print()

    if stable_muon.empty:
        print(
            "No stable Muon regime met the "
            "current definition."
        )
    else:
        best = stable_muon.iloc[0]

        print(
            "Fastest stable Muon regime: "
            f"{best['run']}"
        )

        print(
            "Sustained 95% test step: "
            f"{int(best['sustained_95_test_step'])}"
        )

        print(
            "Speedup versus AdamW: "
            f"{best['speedup_vs_adamw']:.3f}x"
        )


if __name__ == "__main__":
    main()

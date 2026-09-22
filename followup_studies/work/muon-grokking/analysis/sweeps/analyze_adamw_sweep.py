from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the AdamW sweep and compare the fastest "
            "AdamW configurations with stabilized Muon."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/adamw_sweep_manifest_seed_0.csv"
        ),
    )

    parser.add_argument(
        "--stable-muon-csv",
        type=Path,
        default=Path(
            "runs/muon_freeze_all_auxiliary_8000_seed_0.csv"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/adamw_sweep_summary.csv"
        ),
    )

    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(
            "runs/adamw_vs_stable_muon.csv"
        ),
    )

    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path(
            "runs/adamw_sweep_analysis.html"
        ),
    )

    return parser.parse_args()


def first_sustained_step(
    frame: pd.DataFrame,
    field: str,
    threshold: float,
    consecutive: int = 5,
) -> int | None:
    frame = frame.sort_values("step")
    values = frame[field].to_numpy()
    steps = frame["step"].to_numpy()

    for index in range(
        len(values) - consecutive + 1
    ):
        if (
            values[index:index + consecutive]
            >= threshold
        ).all():
            return int(steps[index])

    return None


def first_step(
    frame: pd.DataFrame,
    field: str,
    threshold: float,
) -> int | None:
    selected = frame[
        frame[field] >= threshold
    ]

    if selected.empty:
        return None

    return int(selected.iloc[0]["step"])


def summarize_training_run(
    frame: pd.DataFrame,
) -> dict[str, object]:
    frame = frame.sort_values("step")

    sustained_train = first_sustained_step(
        frame,
        "train_accuracy",
        0.999,
    )
    sustained_test = first_sustained_step(
        frame,
        "test_accuracy",
        0.95,
    )

    if sustained_test is None:
        post = frame.iloc[0:0]
        minimum_post = None
        below_95_count = None
        below_90_count = None
    else:
        post = frame[
            frame["step"] >= sustained_test
        ]
        minimum_post = float(
            post["test_accuracy"].min()
        )
        below_95_count = int(
            (post["test_accuracy"] < 0.95).sum()
        )
        below_90_count = int(
            (post["test_accuracy"] < 0.90).sum()
        )

    return {
        "sustained_99p9_train_step": sustained_train,
        "sustained_95_test_step": sustained_test,
        "memorization_plateau_steps": (
            None
            if (
                sustained_train is None
                or sustained_test is None
            )
            else sustained_test - sustained_train
        ),
        "first_99_test_step": first_step(
            frame,
            "test_accuracy",
            0.99,
        ),
        "first_100_test_step": first_step(
            frame,
            "test_accuracy",
            1.0,
        ),
        "maximum_test_accuracy": float(
            frame["test_accuracy"].max()
        ),
        "final_test_accuracy": float(
            frame.iloc[-1]["test_accuracy"]
        ),
        "minimum_post_grokking_test_accuracy": (
            minimum_post
        ),
        "post_grokking_below_95_count": (
            below_95_count
        ),
        "post_grokking_below_90_count": (
            below_90_count
        ),
        "stable_after_grokking": (
            False
            if minimum_post is None
            else minimum_post >= 0.95
        ),
    }


def sortable_step(value: object) -> float:
    if pd.isna(value):
        return float("inf")
    return float(value)


def write_html(
    path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:
        raise RuntimeError(
            "HTML output requires plotly."
        ) from error

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "AdamW sustained 95% test step",
            "AdamW minimum post-grokking accuracy",
            "Best AdamW versus stabilized Muon",
            "Final test accuracy",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    for weight_decay, group in summary.groupby(
        "weight_decay",
        sort=True,
    ):
        group = group.sort_values(
            "learning_rate"
        )

        figure.add_trace(
            go.Scatter(
                x=group["learning_rate"],
                y=group["sustained_95_test_step"],
                mode="lines+markers",
                name=f"AdamW wd={weight_decay}",
                legendgroup=f"wd={weight_decay}",
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=group["learning_rate"],
                y=group[
                    "minimum_post_grokking_test_accuracy"
                ],
                mode="lines+markers",
                name=f"AdamW wd={weight_decay}",
                legendgroup=f"wd={weight_decay}",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    figure.add_trace(
        go.Bar(
            x=comparison["configuration"],
            y=comparison["sustained_95_test_step"],
            name="Sustained 95% step",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=comparison["configuration"],
            y=comparison["final_test_accuracy"],
            name="Final test accuracy",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    figure.update_xaxes(
        type="log",
        title_text="Learning rate",
        row=1,
        col=1,
    )
    figure.update_xaxes(
        type="log",
        title_text="Learning rate",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="Training step",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title_text="Accuracy",
        range=[0, 1.02],
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title_text="Training step",
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title_text="Accuracy",
        range=[0, 1.02],
        row=2,
        col=2,
    )

    figure.update_layout(
        title=(
            "Compact AdamW tuning comparison "
            "against stabilized Muon"
        ),
        height=900,
        hovermode="x unified",
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.write_html(
        path,
        include_plotlyjs="cdn",
    )


def main() -> None:
    args = parse_arguments()

    manifest = pd.read_csv(args.manifest)
    rows: list[dict[str, object]] = []

    for record in manifest.to_dict(
        orient="records"
    ):
        csv_path = Path(record["csv_path"])

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing sweep result: {csv_path}"
            )

        frame = pd.read_csv(csv_path)
        rows.append(
            {
                **record,
                **summarize_training_run(frame),
            }
        )

    summary = pd.DataFrame(rows)
    summary["selection_step"] = summary[
        "sustained_95_test_step"
    ].map(sortable_step)

    reached = summary[
        summary["sustained_95_test_step"].notna()
    ].copy()

    if reached.empty:
        raise RuntimeError(
            "No AdamW configuration reached sustained "
            "95% test accuracy."
        )

    best_overall = reached.sort_values(
        [
            "selection_step",
            "first_99_test_step",
            "minimum_post_grokking_test_accuracy",
        ],
        ascending=[True, True, False],
    ).iloc[0]

    stable_candidates = reached[
        reached["stable_after_grokking"]
    ].copy()

    if stable_candidates.empty:
        best_stable = None
    else:
        best_stable = stable_candidates.sort_values(
            [
                "selection_step",
                "first_99_test_step",
                "final_test_accuracy",
            ],
            ascending=[True, True, False],
        ).iloc[0]

    if not args.stable_muon_csv.is_file():
        raise FileNotFoundError(
            "Stable Muon CSV not found: "
            f"{args.stable_muon_csv}"
        )

    stable_muon_frame = pd.read_csv(
        args.stable_muon_csv
    )
    stable_muon_metrics = summarize_training_run(
        stable_muon_frame
    )

    comparison_rows = [
        {
            "configuration": (
                "Fastest AdamW"
            ),
            "optimizer": "AdamW",
            "learning_rate": float(
                best_overall["learning_rate"]
            ),
            "weight_decay": float(
                best_overall["weight_decay"]
            ),
            **{
                key: best_overall[key]
                for key in (
                    "sustained_99p9_train_step",
                    "sustained_95_test_step",
                    "memorization_plateau_steps",
                    "first_99_test_step",
                    "first_100_test_step",
                    "maximum_test_accuracy",
                    "final_test_accuracy",
                    "minimum_post_grokking_test_accuracy",
                    "post_grokking_below_95_count",
                    "post_grokking_below_90_count",
                    "stable_after_grokking",
                )
            },
        }
    ]

    if best_stable is not None:
        comparison_rows.append(
            {
                "configuration": (
                    "Fastest stable AdamW"
                ),
                "optimizer": "AdamW",
                "learning_rate": float(
                    best_stable["learning_rate"]
                ),
                "weight_decay": float(
                    best_stable["weight_decay"]
                ),
                **{
                    key: best_stable[key]
                    for key in (
                        "sustained_99p9_train_step",
                        "sustained_95_test_step",
                        "memorization_plateau_steps",
                        "first_99_test_step",
                        "first_100_test_step",
                        "maximum_test_accuracy",
                        "final_test_accuracy",
                        "minimum_post_grokking_test_accuracy",
                        "post_grokking_below_95_count",
                        "post_grokking_below_90_count",
                        "stable_after_grokking",
                    )
                },
            }
        )

    comparison_rows.append(
        {
            "configuration": "Stabilized Muon",
            "optimizer": "Muon",
            "learning_rate": None,
            "weight_decay": None,
            **stable_muon_metrics,
        }
    )

    comparison = pd.DataFrame(
        comparison_rows
    )

    stable_muon_step = stable_muon_metrics[
        "sustained_95_test_step"
    ]

    comparison[
        "speed_relative_to_stable_muon"
    ] = (
        stable_muon_step
        / comparison[
            "sustained_95_test_step"
        ]
    )

    summary = summary.drop(
        columns=["selection_step"]
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        args.summary_output,
        index=False,
    )
    comparison.to_csv(
        args.comparison_output,
        index=False,
    )

    write_html(
        args.html_output,
        summary,
        comparison,
    )

    print()
    print("AdamW sweep:")
    print(
        summary[
            [
                "learning_rate",
                "weight_decay",
                "sustained_95_test_step",
                "first_99_test_step",
                "minimum_post_grokking_test_accuracy",
                "final_test_accuracy",
                "stable_after_grokking",
            ]
        ]
        .sort_values(
            [
                "sustained_95_test_step",
                "learning_rate",
                "weight_decay",
            ],
            na_position="last",
        )
        .to_string(index=False)
    )

    print()
    print("Comparison:")
    print(
        comparison.to_string(index=False)
    )

    print()
    print(
        f"Saved sweep summary: {args.summary_output}"
    )
    print(
        f"Saved comparison: {args.comparison_output}"
    )
    print(
        f"Saved HTML: {args.html_output}"
    )


if __name__ == "__main__":
    main()

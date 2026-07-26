from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REGIME_ORDER = {
    "adamw": 0,
    "muon": 1,
    "stable_muon": 2,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the controlled depth sweep using sustained "
            "generalization and post-grokking stability."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "runs/depth_sweep_v2_manifest_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/depth_sweep_v2_summary_seed_0.csv"
        ),
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path(
            "runs/depth_sweep_v2_comparison_seed_0.csv"
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

    for index in range(len(values) - consecutive + 1):
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
    selected = frame[frame[field] >= threshold]
    if selected.empty:
        return None
    return int(selected.iloc[0]["step"])


def summarize_run(
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
        below_95 = None
        below_90 = None
        grokking_elapsed_seconds = None
    else:
        post = frame[frame["step"] >= sustained_test]
        minimum_post = float(
            post["test_accuracy"].min()
        )
        below_95 = int(
            (post["test_accuracy"] < 0.95).sum()
        )
        below_90 = int(
            (post["test_accuracy"] < 0.90).sum()
        )

        grokking_row = frame[
            frame["step"] == sustained_test
        ]
        if (
            "elapsed_seconds" in frame.columns
            and not grokking_row.empty
        ):
            grokking_elapsed_seconds = float(
                grokking_row.iloc[0]["elapsed_seconds"]
            )
        else:
            grokking_elapsed_seconds = None

    actual_freeze_step = None
    if "auxiliary_frozen" in frame.columns:
        frozen = frame[frame["auxiliary_frozen"] == 1]
        if not frozen.empty:
            actual_freeze_step = int(
                frozen.iloc[0]["step"]
            )

    final_elapsed_seconds = None
    if "elapsed_seconds" in frame.columns:
        final_elapsed_seconds = float(
            frame.iloc[-1]["elapsed_seconds"]
        )

    return {
        "sustained_99p9_train_step": sustained_train,
        "sustained_95_test_step": sustained_test,
        "memorization_plateau_steps": (
            None
            if sustained_train is None
            or sustained_test is None
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
        "post_grokking_below_95_count": below_95,
        "post_grokking_below_90_count": below_90,
        "stable_after_grokking": (
            False
            if minimum_post is None
            else minimum_post >= 0.95
        ),
        "actual_freeze_step": actual_freeze_step,
        "grokking_elapsed_seconds": (
            grokking_elapsed_seconds
        ),
        "final_elapsed_seconds": (
            final_elapsed_seconds
        ),
    }


def main() -> None:
    args = parse_arguments()
    manifest = pd.read_csv(args.manifest)

    rows: list[dict[str, object]] = []

    for record in manifest.to_dict(orient="records"):
        csv_path = Path(record["csv_path"])
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing depth result: {csv_path}"
            )

        frame = pd.read_csv(csv_path)
        rows.append(
            {
                **record,
                **summarize_run(frame),
            }
        )

    summary = pd.DataFrame(rows)
    summary["regime_order"] = summary["regime"].map(
        REGIME_ORDER
    )
    summary = summary.sort_values(
        ["depth", "regime_order"]
    ).drop(columns=["regime_order"])

    comparison_rows: list[dict[str, object]] = []

    for depth, group in summary.groupby(
        "depth",
        sort=True,
    ):
        by_regime = group.set_index("regime")
        adamw_step = by_regime.loc[
            "adamw",
            "sustained_95_test_step",
        ]

        for regime in ["muon", "stable_muon"]:
            regime_step = by_regime.loc[
                regime,
                "sustained_95_test_step",
            ]

            if pd.isna(adamw_step):
                speedup = None
                note = (
                    "AdamW did not reach sustained 95%; "
                    "speedup is right-censored."
                )
            elif pd.isna(regime_step):
                speedup = None
                note = (
                    f"{regime} did not reach sustained 95%."
                )
            else:
                speedup = float(adamw_step) / float(
                    regime_step
                )
                note = ""

            comparison_rows.append(
                {
                    "depth": int(depth),
                    "comparison": (
                        f"{regime}_vs_adamw"
                    ),
                    "adamw_grokking_step": (
                        adamw_step
                    ),
                    "comparison_grokking_step": (
                        regime_step
                    ),
                    "speedup_over_adamw": speedup,
                    "note": note,
                }
            )

    depth_one = summary[
        summary["depth"] == 1
    ].set_index("regime")

    for record in summary.to_dict(orient="records"):
        regime = record["regime"]
        depth = int(record["depth"])
        step = record["sustained_95_test_step"]
        baseline_step = depth_one.loc[
            regime,
            "sustained_95_test_step",
        ]

        if pd.isna(step) or pd.isna(baseline_step):
            scaling = None
        else:
            scaling = float(step) / float(
                baseline_step
            )

        comparison_rows.append(
            {
                "depth": depth,
                "comparison": (
                    f"{regime}_depth_scaling"
                ),
                "adamw_grokking_step": None,
                "comparison_grokking_step": step,
                "speedup_over_adamw": None,
                "depth_scaling_vs_depth_one": scaling,
                "note": "",
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(args.summary_output, index=False)
    comparison.to_csv(
        args.comparison_output,
        index=False,
    )

    display_columns = [
        "depth",
        "regime",
        "sustained_95_test_step",
        "first_99_test_step",
        "minimum_post_grokking_test_accuracy",
        "final_test_accuracy",
        "stable_after_grokking",
        "actual_freeze_step",
    ]

    print()
    print("Controlled depth sweep:")
    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print()
    print("Muon speedup over AdamW by depth:")
    speedups = comparison[
        comparison["comparison"].isin(
            [
                "muon_vs_adamw",
                "stable_muon_vs_adamw",
            ]
        )
    ]
    print(speedups.to_string(index=False))

    print()
    print(f"Saved summary: {args.summary_output}")
    print(
        "Saved comparison: "
        f"{args.comparison_output}"
    )


if __name__ == "__main__":
    main()

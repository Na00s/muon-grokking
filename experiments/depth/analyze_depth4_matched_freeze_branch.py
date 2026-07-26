from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the exact matched depth-four "
            "control-versus-freeze intervention."
        )
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help=(
            "Metadata JSON. When omitted, the newest matching "
            "metadata file in runs/ is used."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/depth4_matched_freeze_"
            "summary_seed_0.csv"
        ),
    )
    return parser.parse_args()


def newest_metadata() -> Path:
    matches = sorted(
        Path("runs").glob(
            "depth4_matched_freeze_seed_*_"
            "from_*_metadata.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError(
            "No matched depth-four metadata file was found."
        )
    return matches[-1]


def longest_true_run(values: pd.Series) -> int:
    longest = 0
    current = 0

    for value in values.astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def summarize(
    frame: pd.DataFrame,
) -> dict[str, object]:
    frame = frame.sort_values("step")
    post = frame.iloc[1:].copy()

    if post.empty:
        raise ValueError(
            "Branch CSV contains no post-intervention rows."
        )

    below_95 = post["test_accuracy"] < 0.95
    below_90 = post["test_accuracy"] < 0.90

    return {
        "branch_step": int(
            frame.iloc[0]["branch_step"]
        ),
        "initial_test_accuracy": float(
            frame.iloc[0]["test_accuracy"]
        ),
        "minimum_post_branch_test_accuracy": float(
            post["test_accuracy"].min()
        ),
        "maximum_post_branch_test_accuracy": float(
            post["test_accuracy"].max()
        ),
        "mean_post_branch_test_accuracy": float(
            post["test_accuracy"].mean()
        ),
        "final_test_accuracy": float(
            post.iloc[-1]["test_accuracy"]
        ),
        "post_branch_below_95_count": int(
            below_95.sum()
        ),
        "post_branch_below_90_count": int(
            below_90.sum()
        ),
        "post_branch_below_95_fraction": float(
            below_95.mean()
        ),
        "post_branch_below_90_fraction": float(
            below_90.mean()
        ),
        "longest_below_95_evaluation_run": (
            longest_true_run(below_95)
        ),
        "longest_below_90_evaluation_run": (
            longest_true_run(below_90)
        ),
        "stable_above_95_after_branch": bool(
            not below_95.any()
        ),
        "final_auxiliary_parameter_norm": float(
            post.iloc[-1][
                "auxiliary_parameter_norm"
            ]
        ),
        "final_unembedding_parameter_norm": float(
            post.iloc[-1][
                "unembedding_parameter_norm"
            ]
        ),
    }


def main() -> None:
    args = parse_arguments()
    metadata_path = (
        args.metadata
        if args.metadata is not None
        else newest_metadata()
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    rows = []
    frames = {}

    for branch in ("control", "freeze"):
        csv_path = Path(
            metadata[f"{branch}_csv"]
        )
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing branch CSV: {csv_path}"
            )

        frame = pd.read_csv(csv_path)
        frames[branch] = frame
        rows.append(
            {
                "branch": branch,
                **summarize(frame),
            }
        )

    summary = pd.DataFrame(rows)
    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary.to_csv(
        args.summary_output,
        index=False,
    )

    control = summary[
        summary["branch"] == "control"
    ].iloc[0]
    freeze = summary[
        summary["branch"] == "freeze"
    ].iloc[0]

    merged = frames["control"][
        ["step", "test_accuracy"]
    ].merge(
        frames["freeze"][
            ["step", "test_accuracy"]
        ],
        on="step",
        suffixes=(
            "_control",
            "_freeze",
        ),
        validate="one_to_one",
    )
    post_merged = merged.iloc[1:]

    mean_paired_advantage = float(
        (
            post_merged[
                "test_accuracy_freeze"
            ]
            - post_merged[
                "test_accuracy_control"
            ]
        ).mean()
    )
    final_paired_advantage = float(
        freeze["final_test_accuracy"]
        - control["final_test_accuracy"]
    )
    minimum_paired_advantage = float(
        freeze[
            "minimum_post_branch_test_accuracy"
        ]
        - control[
            "minimum_post_branch_test_accuracy"
        ]
    )

    print()
    print("Matched depth-four freeze intervention:")
    print(summary.to_string(index=False))

    print()
    print("Causal comparison from the identical checkpoint:")
    print(
        "Mean paired test-accuracy advantage "
        f"of freezing: {mean_paired_advantage:+.6f}"
    )
    print(
        "Final test-accuracy advantage "
        f"of freezing: {final_paired_advantage:+.6f}"
    )
    print(
        "Minimum-accuracy advantage "
        f"of freezing: {minimum_paired_advantage:+.6f}"
    )
    print(
        "Control evaluations below 90%: "
        f"{int(control['post_branch_below_90_count'])}"
    )
    print(
        "Freeze evaluations below 90%: "
        f"{int(freeze['post_branch_below_90_count'])}"
    )
    print(
        "Exact branch parameter difference: "
        f"{metadata['branch_parameter_max_abs_difference']}"
    )
    print()
    print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()

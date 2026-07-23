from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare auxiliary-component freeze branches."
        )
    )

    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/auxiliary_component_branches.html"
        ),
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "runs/auxiliary_component_branches_summary.csv"
        ),
    )

    return parser.parse_args()


def first_collapse_step(
    frame: pd.DataFrame,
) -> int | None:
    collapse_rows = frame.loc[
        frame["collapse_detected"] == 1,
        "step",
    ]

    if collapse_rows.empty:
        return None

    return int(
        collapse_rows.iloc[0]
    )


def main() -> None:
    args = parse_arguments()

    frames: list[tuple[str, pd.DataFrame]] = []

    for path in args.files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CSV: {path}"
            )

        frame = pd.read_csv(path)

        label = str(
            frame["branch"].iloc[0]
        )

        frames.append(
            (label, frame)
        )

    figure = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=(
            "Test accuracy",
            "Train accuracy",
            "Hidden-matrix displacement",
            "Token-embedding displacement",
            "Position-embedding displacement",
            "Unembedding displacement",
            "Other auxiliary displacement",
            "Auxiliary gradient norm",
        ),
    )

    subplot_columns = [
        ("test_accuracy", 1, "Accuracy"),
        ("train_accuracy", 2, "Accuracy"),
        ("hidden_delta_norm", 3, "L2 displacement"),
        ("token_embedding_delta_norm", 4, "L2 displacement"),
        ("position_embedding_delta_norm", 5, "L2 displacement"),
        ("unembedding_delta_norm", 6, "L2 displacement"),
        ("other_auxiliary_delta_norm", 7, "L2 displacement"),
        ("auxiliary_gradient_norm", 8, "Gradient norm"),
    ]

    for label, frame in frames:
        for column, row, _ in subplot_columns:
            figure.add_trace(
                go.Scatter(
                    x=frame["step"],
                    y=frame[column],
                    mode="lines",
                    name=f"{label}: {column}",
                    legendgroup=label,
                    showlegend=(
                        column == "test_accuracy"
                    ),
                ),
                row=row,
                col=1,
            )

    figure.update_yaxes(
        range=[-0.02, 1.02],
        row=1,
        col=1,
    )

    figure.update_yaxes(
        range=[-0.02, 1.02],
        row=2,
        col=1,
    )

    for _, row, axis_title in subplot_columns:
        figure.update_yaxes(
            title_text=axis_title,
            row=row,
            col=1,
        )

    figure.update_yaxes(
        type="log",
        row=8,
        col=1,
    )

    figure.update_xaxes(
        title_text="Training step",
        row=8,
        col=1,
    )

    figure.update_layout(
        title=(
            "Auxiliary-component ablations around "
            "the first Muon collapse"
        ),
        template="plotly_white",
        height=1800,
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

    summary_rows = []

    for label, frame in frames:
        summary_rows.append(
            {
                "branch": label,
                "frozen_group": str(
                    frame["frozen_group"].iloc[0]
                ),
                "first_collapse_step": (
                    first_collapse_step(frame)
                ),
                "collapse_evaluations": int(
                    frame["collapse_detected"].sum()
                ),
                "minimum_train_accuracy": float(
                    frame["train_accuracy"].min()
                ),
                "minimum_test_accuracy": float(
                    frame["test_accuracy"].min()
                ),
                "final_train_accuracy": float(
                    frame.iloc[-1]["train_accuracy"]
                ),
                "final_test_accuracy": float(
                    frame.iloc[-1]["test_accuracy"]
                ),
                "maximum_auxiliary_gradient_norm": float(
                    frame["auxiliary_gradient_norm"].max()
                ),
                "final_hidden_delta_norm": float(
                    frame.iloc[-1]["hidden_delta_norm"]
                ),
                "final_token_embedding_delta_norm": float(
                    frame.iloc[-1][
                        "token_embedding_delta_norm"
                    ]
                ),
                "final_position_embedding_delta_norm": float(
                    frame.iloc[-1][
                        "position_embedding_delta_norm"
                    ]
                ),
                "final_unembedding_delta_norm": float(
                    frame.iloc[-1][
                        "unembedding_delta_norm"
                    ]
                ),
                "final_other_auxiliary_delta_norm": float(
                    frame.iloc[-1][
                        "other_auxiliary_delta_norm"
                    ]
                ),
            }
        )

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


if __name__ == "__main__":
    main()

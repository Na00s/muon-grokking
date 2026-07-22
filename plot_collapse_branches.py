from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the three checkpoint-branch experiments."
        )
    )

    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="Branch CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "collapse_branch_comparison.html"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    frames: list[tuple[str, pd.DataFrame]] = []

    for path in args.files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing CSV: {path}"
            )

        frame = pd.read_csv(path)

        if "branch" not in frame.columns:
            raise ValueError(
                f"{path} is missing the branch column."
            )

        label = str(
            frame["branch"].iloc[0]
        )

        frames.append(
            (label, frame)
        )

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Test accuracy",
            "Train accuracy",
            "Parameter displacement from branch point",
            "Gradient norms",
        ),
    )

    for label, frame in frames:
        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["test_accuracy"],
                mode="lines",
                name=f"{label}: test",
            ),
            row=1,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["train_accuracy"],
                mode="lines",
                name=f"{label}: train",
            ),
            row=2,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["hidden_delta_norm"],
                mode="lines",
                name=f"{label}: hidden delta",
            ),
            row=3,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["auxiliary_delta_norm"],
                mode="lines",
                line={"dash": "dash"},
                name=f"{label}: auxiliary delta",
            ),
            row=3,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["hidden_gradient_norm"],
                mode="lines",
                name=f"{label}: hidden gradient",
            ),
            row=4,
            col=1,
        )

        figure.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["auxiliary_gradient_norm"],
                mode="lines",
                line={"dash": "dash"},
                name=f"{label}: auxiliary gradient",
            ),
            row=4,
            col=1,
        )

    figure.update_yaxes(
        title_text="Accuracy",
        range=[-0.02, 1.02],
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Accuracy",
        range=[-0.02, 1.02],
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="L2 displacement",
        row=3,
        col=1,
    )

    figure.update_yaxes(
        title_text="Gradient norm",
        type="log",
        row=4,
        col=1,
    )

    figure.update_xaxes(
        title_text="Training step",
        row=4,
        col=1,
    )

    figure.update_layout(
        title=(
            "Causal checkpoint branches around "
            "the first Muon collapse"
        ),
        template="plotly_white",
        height=1200,
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
    print()

    for label, frame in frames:
        collapses = frame.loc[
            frame["collapse_detected"] == 1
        ]

        minimum_train = float(
            frame["train_accuracy"].min()
        )

        minimum_test = float(
            frame["test_accuracy"].min()
        )

        print(
            f"{label}: "
            f"minimum train accuracy={minimum_train:.4f}, "
            f"minimum test accuracy={minimum_test:.4f}, "
            f"collapse evaluations={len(collapses)}"
        )


if __name__ == "__main__":
    main()

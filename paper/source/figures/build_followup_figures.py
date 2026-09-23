"""Plot the recorded follow-up results; this script performs no model fitting.

Run with the research Python environment from any working directory. The default data root is the repository's followup_studies directory. A standalone
source archive requires --workspace /path/to/followup_studies; that directory must
contain work/. The default output directory is the paper source directory.
The adjacent swaps, selected decoder snapshots, and matched arithmetic branches
reuse the same five seeds. No panel reports independent replicate counts for
multiple interventions on the same seed, and no statistical intervals are fitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "plot_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    candidate = HERE.parents[2] / "followup_studies"
    parser.add_argument("--workspace", type=Path, default=candidate if (candidate / "work").is_dir() else None,
                        help="Released followup_studies directory containing work/; required outside the repository.")
    parser.add_argument("--output", type=Path, default=HERE.parent,
                        help="Paper source directory containing the six original figures.")
    parser.add_argument("--only-mechanism", action="store_true",
                        help="Regenerate the main mechanism figure while preserving the RMS figure files.")
    args = parser.parse_args()
    if args.workspace is None:
        parser.error("Supply --workspace /path/to/followup_studies (the directory containing work/).")
    if not (args.workspace / "work").is_dir():
        parser.error("--workspace must contain the released work/ results directory.")
    root, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(relative):
        path = root / relative
        data = path.read_bytes()
        sources[relative] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    existing_figures = sorted(output.glob("figure[1-6]_*.pdf"))
    assert len(existing_figures) == 6
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in existing_figures}
    preserved_figures = sorted(p for p in output.iterdir()
                              if p.suffix in {".pdf", ".png"}
                              and p.stem != "followup_mechanism") if args.only_mechanism else []
    preserved_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved_figures}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.linewidth": .65,
        "xtick.major.width": .65, "ytick.major.width": .65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.axisbelow": True,
        "mathtext.fontset": "dejavusans",
    })
    gray, blue, orange, red = "#747474", "#0072B2", "#D55E00", "#AE3047"

    def clean(ax, *, accuracy=True):
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, pad=2)
        ax.grid(axis="y", color="#E5E5E5", linewidth=.5)
        if accuracy:
            ax.set_ylim(-3, 106)
            ax.set_yticks([0, 25, 50, 75, 100])

    def save(fig, stem):
        if args.only_mechanism and stem != "followup_mechanism":
            plt.close(fig)
            return
        # Fixed 5.5-inch width matches this ICLR template's \textwidth.
        fig.savefig(output / (stem + ".pdf"), metadata={
            "Title": stem.replace("_", " "), "Author": "",
            "Subject": "Recorded experimental results; reproducibility manifest accompanies plotting script",
            "CreationDate": None, "ModDate": None,
        })
        fig.savefig(output / (stem + ".png"), dpi=300)
        plt.close(fig)

    pairs = ["adjacent_seed0", "seed1_adjacent_event", "seed2_adjacent_event",
             "seed3_adjacent_event", "adjacent_seed4"]
    all_swaps = read("work/basis_study/bilinear/summary.json")
    swaps = []
    for pair in pairs:
        matched = [row for row in all_swaps if row["pair"] == pair]
        assert len(matched) == 1
        assert matched[0]["heldout_examples"] == 8939
        swaps.append(matched[0])
    assert len(swaps) == 5
    swap_keys = ["preceding_accuracy", "feature_change_only_accuracy",
                 "readout_change_only_accuracy", "actual_later_accuracy"]
    swap_values = np.array([[row[key] for key in swap_keys] for row in swaps]) * 100
    assert swap_values.shape == (5, 4)

    probe_folders = [
        "work/experiments/seed0_dense_event", "work/basis_study/seed1_first_event",
        "work/basis_study/seed2_first_event", "work/basis_study/seed3_peak_event",
        "work/experiments/seed4_peak_event",
    ]
    probe_steps, native_probe, fresh_probe = [], [], []
    for seed, folder in enumerate(probe_folders):
        metadata = read(folder + "/metadata.json")
        decoder = read(folder + "/decoder_whitened.json")["cv_selected"]["collapsed"]
        assert metadata["seed"] == seed
        assert decoder["refit"]["converged"]
        assert decoder["split"]["original_train_count"] == 3830
        assert decoder["split"]["heldout_count"] == 8939
        assert decoder["refit"]["feature_transform"]["centered"] is False
        probe_steps.append(metadata["collapsed_step"])
        native_probe.append(metadata["native_collapsed"]["heldout"]["accuracy"] * 100)
        fresh_probe.append(decoder["classification"]["heldout"]["accuracy"] * 100)
    assert probe_steps == [17494, 17721, 18069, 16309, 16060]
    assert len(native_probe) == len(fresh_probe) == 5

    results = read("work/causal_study/analysis/results.json")
    primary = sorted(results["primary"], key=lambda item: item["seed"])
    assert results["complete"] and [p["seed"] for p in primary] == list(range(5))
    assert all(p["accurate"]["start"] == 6000 and p["accurate"]["end"] == 30000
               and p["accurate"]["dense_endpoint_validated"] for p in primary)
    original_events = [6000 <= p["stock_event_step"] <= 30000
                       and p["stock_event_train_accuracy"] < .9
                       and p["stock_event_test_accuracy"] < .9 for p in primary]
    extended = read("work/long_horizon/summary.json")
    extended_addition = sorted([r for r in extended["records"] if r["operation"] == "addition"], key=lambda r: r["seed"])
    assert extended["status"] == "completed" and [r["seed"] for r in extended_addition] == list(range(5))
    assert all(r["arithmetic_start_step"] == 6000 and r["end_step"] == 100000 for r in extended_addition)
    accurate_events = [r["first_joint_failure_step"] is not None for r in extended_addition]
    reduced_lr = read("work/lr_reduction/summary.json")
    reduced_lr_protocol = read("work/lr_reduction/protocol.json")
    reduced_lr_intervention = reduced_lr_protocol["intervention"]
    assert reduced_lr_intervention["hidden_muon_learning_rate"] == 0.003
    assert reduced_lr_intervention["original_hidden_muon_learning_rate"] == 0.03
    assert reduced_lr_intervention["cross_entropy"] == "stock"
    assert reduced_lr_intervention["freeze_parameters"] is False
    assert reduced_lr_intervention["reset_optimizer_buffers"] is False
    assert reduced_lr_intervention["embedding_learning_rate"] == 0.001
    assert reduced_lr_intervention["readout_learning_rate"] == 0.00025
    reduced_lr_records = sorted(reduced_lr["records"], key=lambda r: r["seed"])
    assert reduced_lr["status"] == "completed_and_verified"
    assert reduced_lr["seeds"] == 5 and reduced_lr["total_updates"] == 470000
    assert [r["seed"] for r in reduced_lr_records] == list(range(5))
    assert all(r["start_step"] == 6000 and r["end_step"] == 100000
               and r["updates"] == 94000 and r["train_states"] == 94001
               for r in reduced_lr_records)
    reduced_lr_events = [r["first_joint_failure_step"] is not None for r in reduced_lr_records]
    assert all(event == r["joint_failure"] == (r["joint_below90_states"] > 0)
               for event, r in zip(reduced_lr_events, reduced_lr_records))
    incidence = [sum(original_events), sum(reduced_lr_events), sum(accurate_events)]
    assert incidence == [5, 5, 0]
    assert reduced_lr["seeds_with_joint_failure"] == incidence[1]
    specificity = results["specificity"]
    assert len(specificity) == 5
    for row in specificity:
        for key in ["accurate", "target_repair", "row_projection"]:
            assert row[key]["start"] == 15000 and row[key]["end"] == 20000
            assert row[key]["event"] is False

    # Preserve the 1.479-inch plot height while removing unused vertical space.
    fig = plt.figure(figsize=(5.5, 2.55))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.15, 1.45],
                           left=.085, right=.985, bottom=.80 / 2.55,
                           top=(.80 + 1.479) / 2.55, wspace=.30)
    axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    a, b, c = axes
    for ax in [a, b]:
        clean(ax)
        ax.set_xlim(-.50, 4.50)
        ax.set_xticks(range(5))
        ax.set_xlabel("Seed", labelpad=3)
    b.set_yticklabels([])
    a.set_ylabel("Test accuracy (%)", labelpad=3)
    a.set_title("(a) Adjacent swaps", loc="left", pad=10)
    colors = [gray, blue, orange, red]
    markers = ["o", "^", "D", "x"]
    labels = [r"$W_0h_0$", r"$W_0h_1$", r"$W_1h_0$", r"$W_1h_1$"]
    for j, (color, marker, label) in enumerate(zip(colors, markers, labels)):
        a.plot(np.arange(5) + (j - 1.5) * .14, swap_values[:, j],
               linestyle="none", marker=marker, markersize=4,
               markerfacecolor="white" if j == 0 else color,
               markeredgewidth=.85, color=color, label=label)
    a.legend(loc="upper center", bbox_to_anchor=(.5, -.29), ncol=2,
             frameon=False, columnspacing=.45, handletextpad=.25, handlelength=1.0,
             borderaxespad=0, labelspacing=.5)

    b.set_title("(b) Fresh decoders", loc="left", pad=10)
    b.vlines(range(5), native_probe, fresh_probe, color="#B9B9B9", linewidth=.8)
    b.plot(range(5), native_probe, linestyle="none", marker="x", markersize=4.5,
           color=red, label="Native head")
    b.plot(range(5), fresh_probe, linestyle="none", marker="^", markersize=4,
           color=blue, label="Fresh decoder")
    b.legend(loc="upper center", bbox_to_anchor=(.5, -.29), frameon=False,
             handletextpad=.35, handlelength=1.0, borderaxespad=0, labelspacing=.5)

    clean(c, accuracy=False)
    c.set_title("(c) Joint failures", loc="left", pad=10)
    c.bar([0, 1, 2], incidence, width=.58, color=[red, orange, blue], linewidth=.7,
          edgecolor=[red, orange, blue], alpha=.9)
    c.plot([2], [0], marker="o", markersize=4, color=blue, clip_on=False)
    c.set_ylim(0, 5.65)
    c.set_yticks([0, 1, 2, 3, 4, 5])
    c.set_ylabel("Seeds with failure", labelpad=2)
    c.set_xticks([0, 1, 2], ["Original\nCE", "Hidden\nLR / 10", "Accurate\nCE"])
    c.tick_params(axis="x", length=0, pad=5)
    for xpos, value in enumerate(incidence):
        c.text(xpos, value + .20, f"{value}/5", ha="center", va="bottom", fontsize=8)
    c.text(.5, -.30, "Steps 6,000\nto 100,000", transform=c.transAxes,
           ha="center", va="top", fontsize=8, linespacing=1.3)
    save(fig, "followup_mechanism")

    generality_names = ["subtraction_stock", "rms_stock", "rms_accurate"]
    generality_titles = ["(a) Subtraction, stock CE", "(b) RMS, stock CE", "(c) RMS, accurate CE"]
    parameter_masks = ["000", "001", "010", "111"]
    raw_generality = []
    for name in generality_names:
        data = read(f"work/causal_study/generality/{name}/event_analysis.json")
        assert data["exact_next_update_replay"] is True
        assert data["post_step"] == data["pre_step"] + 1
        hybrids = {row["mask"]: row for row in data["parameter_hybrids"]}
        assert len(hybrids) == 8
        assert hybrids["001"]["updated_groups"] == ["readout"]
        assert hybrids["010"]["updated_groups"] == ["embeddings"]
        values = [hybrids[k]["metrics"]["heldout"]["accuracy"] * 100 for k in parameter_masks]
        raw_generality.append({"condition": name, "pre_step": data["pre_step"],
                               "post_step": data["post_step"], "test_accuracy": values})
    assert [r["post_step"] for r in raw_generality] == [15281, 18515, 28495]

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.65), sharey=True)
    fig.subplots_adjust(left=.09, right=.985, bottom=.32, top=.76, wspace=.20)
    for ax, title, data in zip(axes, generality_titles, raw_generality):
        clean(ax)
        values = data["test_accuracy"]
        ax.bar(range(4), values, width=.62, color=[gray, blue, orange, red],
               linewidth=0, zorder=2)
        ax.set_ylim(0, 115)
        ax.set_xticks(range(4), ["Before", "Head only", "Embed. only", "Full"],
                      rotation=30, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", length=0, pad=5)
        ax.set_title(title, loc="left", pad=24)
        ax.text(.5, 1.075, f'{data["pre_step"]:,} to {data["post_step"]:,}',
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        for xpos, value in enumerate(values):
            ax.text(xpos, value + 2.5, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    axes[0].set_ylabel("Test accuracy (%)", labelpad=3)
    fig.text(.53, .065, "Parameter groups changed in the captured update", ha="center", fontsize=8)
    save(fig, "followup_rms_boundary")

    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in existing_figures}
    assert before == after
    assert preserved_before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in preserved_figures}
    manifest = {
        "scope": "Recorded values only; no new training, fitting, or uncertainty estimation.",
        "figure_width_inches": 5.5, "minimum_font_points": 8,
        "main_figure_height_inches": 2.55, "main_plot_height_inches": 1.479,
        "source_sha256": sources, "original_figure_sha256_unchanged": after,
        "main_panel_a": {"pairs": pairs, "conditions": labels, "test_accuracy_percent": swap_values.tolist()},
        "main_panel_b": {"seeds": list(range(5)), "checkpoint_steps": probe_steps,
                         "native_test_accuracy_percent": native_probe,
                         "fresh_decoder_test_accuracy_percent": fresh_probe},
        "main_panel_c": {"seeds": list(range(5)), "original_failure_count": incidence[0],
                         "reduced_hidden_lr_failure_count": incidence[1],
                         "accurate_failure_count": incidence[2], "source_step": 6000, "end_step": 100000,
                         "reduced_hidden_lr": 0.003, "original_hidden_lr": 0.03,
                         "reduced_lr_updates_per_seed": 94000,
                         "reduced_lr_total_updates": 470000,
                         "reduced_lr_first_failure_steps": [r["first_joint_failure_step"] for r in reduced_lr_records],
                         "original_events_observed_by_step": 30000,
                         "event": "Both train and test accuracy below 90% after grokking.",
                         "caveat": "Paired seeds; original failures were observed by 30,000 under mixed monitoring. Reduced-LR and accurate branches continue through 100,000 with test evaluation every 100 updates and whenever training accuracy falls below 90%. Binary incidence only. Ordinary LR reduction also scales hidden decoupled weight decay."},
        "specificity_confirmed_but_not_plotted": "All three arms have 0/5 events between 15,000 and 20,000.",
        "appendix_parameter_masks": parameter_masks,
        "appendix_first_events": raw_generality,
        "dependency_note": "Main panels reuse five seeds; appendix conditions each have one seed.",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for stem in ["followup_mechanism", "followup_rms_boundary"]
                    for suffix in ["pdf", "png"] for p in [output / f"{stem}.{suffix}"]},
    }
    (HERE / "followup_figure_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": list(manifest["outputs"]), "sources": len(sources),
                      "original_figures_unchanged": before == after, "incidence": incidence,
                      "probe_steps": probe_steps}, indent=2))


if __name__ == "__main__":
    main()

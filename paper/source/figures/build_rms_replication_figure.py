"""Plot verified accurate-CE RMS event swaps and training-only decoder results.

Default: require a complete study/verification.json and study/summary.json.
Use --study /path/to/rms_replication from an editable source archive. The optional
--pilot-only-qa mode reads only the independently audited historical seed-0 event
and requires an explicit output directory outside the manuscript. It never plots
unfinished prospective results. This script performs no fitting or training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rms_figure_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import numpy as np


HERE = Path(__file__).resolve().parent
GROUPS = ["embeddings", "hidden", "readout"]
MASKS = [format(k, "03b") for k in range(8)]
AUDIT_FLAGS = ["passed", "replay_passed", "derivative_arrays_independently_recomputed",
               "decoder_predictions_independently_recomputed", "validation_indices_verified",
               "source_artifact_hashes_verified"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_study(study, pilot_only=False):
    """Require completed audits, check summary identity and exact plotted counts."""
    sources = {}

    def read(relative):
        path = study / relative
        require(path.is_file(), f"Required verified artifact missing: {relative}")
        sources[relative] = sha(path)
        return json.loads(path.read_text())

    protocol = read("protocol.json")
    if pilot_only:
        verification_name = "verification.json" if (study / "verification.json").exists() else "partial_verification.json"
        verification = read(verification_name)
        events = [read("analyses/historical_seed0/summary.json")]
        summary = {"event_summaries": events, "prospective_completed": 0,
                   "prospective_seed_count": len(protocol["prospective_seeds"])}
    else:
        verification = read("verification.json")
        require(verification.get("complete") is True, "Study completion verification must pass.")
        require(not verification.get("remaining"), "Completion audit lists unfinished work.")
        summary = read("summary.json")
        require(not summary.get("remaining"), "Study summary lists unfinished work.")
        expected_seeds = set(protocol["prospective_seeds"])
        runs = summary["runs"]
        require({r["seed"] for r in runs} == expected_seeds and len(runs) == len(expected_seeds),
                "Summary must contain each prospective seed exactly once.")
        require(summary["prospective_completed"] == summary["prospective_seed_count"] == len(expected_seeds),
                "Prospective study remains incomplete.")
        require(runs == verification["runs"], "Run summary differs from completion audit.")
        accepted = {"confirmed_grokking_with_failure", "confirmed_grokking_event_free_completion",
                    "failure_to_confirm_grokking"}
        require(all(r["status"] in accepted for r in runs), "Resolve technical interruptions before plotting.")
        expected_events = {("historical_cpu_pilot", 0)} | {
            ("prospective_cpu", r["seed"]) for r in runs if r["status"] == "confirmed_grokking_with_failure"}
        events = summary["event_summaries"]
        require({(e["cohort"], e["seed"]) for e in events} == expected_events,
                "Captured event coverage differs from completed run outcomes.")
        require(len(events) == len(expected_events), "Duplicate event summaries.")
        require(summary["prospective_failure_count"] == len(expected_events) - 1,
                "Failure incidence disagrees with captured-event summaries.")
    require(verification.get("immutable_source_hashes_verified") is True, "Source verification must pass.")
    require(verification["protocol_sha256"] == sources["protocol.json"], "Protocol hash differs from verification.")
    events = sorted(events, key=lambda e: (e["cohort"] != "historical_cpu_pilot", e["seed"]))
    audits = {(a["cohort"], a["seed"]): a for a in verification["event_audits"]}
    for event in events:
        key = (event["cohort"], event["seed"])
        require(key in audits and all(audits[key].get(k) is True for k in AUDIT_FLAGS),
                f"Event lacks complete independent verification: {key}")
        folder = "analyses/historical_seed0" if key == ("historical_cpu_pilot", 0) else f"analyses/prospective_seed{event['seed']}"
        stored = read(f"{folder}/summary.json")
        require(stored == event, f"Aggregate and per-event summaries differ: {key}")
        require(read(f"{folder}/completion.json").get("passed") is True, f"Event incomplete: {key}")
        replay = read(f"{folder}/replay_verification.json")
        require(replay.get("passed") is True and replay.get("metrics_exact") is True,
                f"Replay verification failed: {key}")
        require(event["exact_replay"] is True and event["failed_step"] == event["previous_step"] + 1,
                f"Expected adjacent exactly replayed event: {key}")
        require([s["mask"] for s in event["swaps"]] == MASKS, f"Expected all eight ordered swaps: {key}")
        require(event["swaps"][0]["metrics"] == event["native_previous"] and
                event["swaps"][-1]["metrics"] == event["native_failed"], f"Swap corners disagree: {key}")
        for swap in event["swaps"]:
            require(swap["group_order"] == GROUPS, f"Unexpected parameter-group order: {key}")
            metric = swap["metrics"]["test"]
            require(metric["accuracy"] == metric["correct"] / metric["total"], f"Swap count mismatch: {key}")
            require(math.isfinite(metric["accuracy"]) and 0 <= metric["accuracy"] <= 1,
                    f"Invalid swap accuracy: {key}")
        for state, result in event["decoders"]["cv_selected"].items():
            metric = result["classification"]["heldout"]
            require(metric["accuracy"] == metric["correct_count"] / metric["n"], f"Decoder count mismatch: {key}")
            require(isinstance(result["refit_converged"], bool), f"Missing convergence status: {key}")
            checked = next(a for a in audits[key]["decoder_checks"] if a["checkpoint"] == state and a["method"] == "cv_selected")
            require(checked["heldout_correct"] == metric["correct_count"] and
                    checked["refit_converged"] == result["refit_converged"], f"Decoder differs from audit: {key}")
    return events, summary, sources


def make_figure(events, output, pilot_only, sources, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.linewidth": .6,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    n = len(events)
    # Exactly 396 PDF points across; no bbox='tight' resizing at save time.
    width, height = 5.5, max(2.45, 1.60 + .255 * n)
    fig = plt.figure(figsize=(width, height))
    top = .82 if pilot_only else .90
    bottom = .39
    heat = fig.add_axes([.125, bottom, .495, top - bottom])
    paired = fig.add_axes([.735, bottom, .25, top - bottom])
    values = np.array([[100 * s["metrics"]["test"]["accuracy"] for s in e["swaps"]] for e in events])
    norm = Normalize(vmin=0, vmax=100)
    cmap = plt.get_cmap("cividis")
    im = heat.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    heat.set_title("(a) Parameter-group swaps", loc="left", pad=8, weight="bold")
    heat.set_xticks(range(8), MASKS)
    heat.set_yticks(range(n), [f"Pilot {e['seed']}" if e["cohort"] == "historical_cpu_pilot" else f"Seed {e['seed']}" for e in events])
    heat.tick_params(length=0, pad=4)
    for r in range(n):
        for c in range(8):
            rgba = cmap(norm(values[r, c]))
            luminance = .2126 * rgba[0] + .7152 * rgba[1] + .0722 * rgba[2]
            heat.text(c, r, f"{values[r, c]:.1f}", ha="center", va="center", fontsize=8,
                      color="white" if luminance < .48 else "#111111")
    heat.set_xticks(np.arange(-.5, 8, 1), minor=True)
    heat.set_yticks(np.arange(-.5, n, 1), minor=True)
    heat.grid(which="minor", color="white", linewidth=.65)
    heat.tick_params(which="minor", bottom=False, left=False)
    for spine in heat.spines.values():
        spine.set_visible(False)
    paired.set_title("(b) Fresh decoder", loc="left", pad=8, weight="bold")
    paired.set_xlim(-3, 103)
    paired.set_ylim(n - .5, -.5)
    paired.set_xticks([0, 50, 100])
    paired.set_yticks(range(n), [])
    paired.set_xlabel("Test accuracy (%)", labelpad=5)
    paired.tick_params(axis="y", length=0)
    paired.tick_params(axis="x", length=3, pad=4)
    paired.spines[["top", "right", "left"]].set_visible(False)
    paired.grid(axis="x", color="#dddddd", linewidth=.55)
    before_color, after_color = "#0072B2", "#D55E00"
    not_converged = False
    plotted = []
    for r, event in enumerate(events):
        decoders = event["decoders"]["cv_selected"]
        before = 100 * decoders["previous"]["classification"]["heldout"]["accuracy"]
        after = 100 * decoders["failed"]["classification"]["heldout"]["accuracy"]
        paired.plot([before, after], [r, r], color="#a0a0a0", linewidth=1, zorder=1)
        for state, value, color, marker in [("previous", before, before_color, "o"), ("failed", after, after_color, "s")]:
            converged = decoders[state]["refit_converged"]
            not_converged |= not converged
            paired.plot(value, r, marker=marker, markersize=5.4,
                        markerfacecolor=color if converged else "white", markeredgecolor=color,
                        markeredgewidth=1, linestyle="none", zorder=3, clip_on=False)
            if not converged:
                paired.plot(value, r, marker="x", color="#111111", markersize=6.8,
                            markeredgewidth=.8, linestyle="none", zorder=4, clip_on=False)
        plotted.append({"cohort": event["cohort"], "seed": event["seed"],
                        "previous_step": event["previous_step"], "failed_step": event["failed_step"],
                        "swap_test_accuracy_percent": values[r].tolist(),
                        "cv_decoder_before_percent": before, "cv_decoder_after_percent": after,
                        "cv_before_refit_converged": decoders["previous"]["refit_converged"],
                        "cv_after_refit_converged": decoders["failed"]["refit_converged"]})
    if n > 1 and events[0]["cohort"] == "historical_cpu_pilot":
        for ax in [heat, paired]:
            ax.axhline(.5, color="#222222", linewidth=.8, linestyle=(0, (2, 2)))
    color_axis = fig.add_axes([.165, .18, .29, .033])
    cb = fig.colorbar(im, cax=color_axis, orientation="horizontal", ticks=[0, 50, 100])
    cb.ax.tick_params(length=2, pad=2, labelsize=8)
    cb.set_label("Test accuracy (%)", fontsize=8, labelpad=2)
    cb.outline.set_linewidth(.5)
    handles = [Line2D([], [], marker="o", color=before_color, linestyle="none", markersize=5, label="Before"),
               Line2D([], [], marker="s", color=after_color, linestyle="none", markersize=5, label="After")]
    if not_converged:
        handles.append(Line2D([], [], marker="x", color="#111111", linestyle="none", markersize=5, label="Refit unconverged"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.735, .203), ncol=1 if not_converged else 2,
               frameon=False, handletextpad=.35, columnspacing=.9, borderpad=0, labelspacing=.4)
    fig.text(.125, .285, "E: embeddings   H: hidden matrices   R: readout", fontsize=8)
    fig.text(.125, .235, "0: before update   1: after update", fontsize=8)
    footer = "Historical CPU pilot only; prospective results are excluded from this QA preview." if pilot_only else "Pilot 0: historical CPU. Other rows: prospective CPU captured failures."
    fig.text(.5, .018, footer, ha="center", fontsize=8)
    if pilot_only:
        fig.text(.5, .968, "PILOT-ONLY QA: verified historical event", ha="center", va="top", fontsize=9, weight="bold", color="#8b2f27")
    fig.canvas.draw()
    # Text extents identify clipping while preserving exact print dimensions.
    renderer = fig.canvas.get_renderer()
    clipped = []
    for item in fig.findobj(matplotlib.text.Text):
        if item.get_visible() and item.get_text():
            bb = item.get_window_extent(renderer)
            if bb.x0 < -.5 or bb.y0 < -.5 or bb.x1 > fig.bbox.width + .5 or bb.y1 > fig.bbox.height + .5:
                clipped.append(item.get_text())
    require(not clipped, "Text outside figure canvas: " + repr(clipped))
    stem = "rms_replication_pilot_only_QA" if pilot_only else "followup_rms_replication"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{stem}.pdf"
    png = output / f"{stem}.png"
    fig.savefig(pdf, metadata={"Title": stem.replace("_", " "), "Author": "",
                              "Subject": "Recorded RMS event swaps and training-only decoder accuracy",
                              "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    swap_rows, decoder_rows = [], []
    for event in events:
        common = {key: event[key] for key in ["cohort", "seed", "previous_step", "failed_step"]}
        for swap in event["swaps"]:
            metric = swap["metrics"]["test"]
            swap_rows.append(dict(common, mask_EHR=swap["mask"], correct=metric["correct"],
                                  total=metric["total"], test_accuracy_percent=100 * metric["accuracy"]))
        for state in ["previous", "failed"]:
            decoder = event["decoders"]["cv_selected"][state]
            metric = decoder["classification"]["heldout"]
            decoder_rows.append(dict(common, checkpoint=state, correct=metric["correct_count"],
                                     total=metric["n"], test_accuracy_percent=100 * metric["accuracy"],
                                     refit_converged=decoder["refit_converged"]))
    csv_paths = []
    for suffix, rows in [("swaps", swap_rows), ("decoders", decoder_rows)]:
        path = output / f"{stem}_{suffix}.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        csv_paths.append(path)
    manifest = {"mode": "PILOT_ONLY_QA" if pilot_only else "COMPLETE_VERIFIED_STUDY",
                "generator": {"path": "figures/build_rms_replication_figure.py", "sha256": sha(Path(__file__))},
                "source_sha256": sources, "figure_width_pdf_points": 396,
                "figure_height_pdf_points": height * 72, "ordinary_text_minimum_points": 8,
                "accuracy_units": "percent", "accuracy_color_scale": [0, 100],
                "group_order": GROUPS, "swap_masks": MASKS,
                "decoder": "CV-selected training-only decoder using actual input after final RMS",
                "convergence_encoding": "Filled markers: converged refit; open marker with x: unconverged refit.",
                "prospective_completed": summary["prospective_completed"],
                "prospective_seed_count": summary["prospective_seed_count"],
                "plotted_events": plotted, "text_outside_canvas": clipped,
                "outputs": {p.name: sha(p) for p in [pdf, png, *csv_paths]}}
    (output / f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, help="RMS study directory containing summary.json and verification.json.")
    parser.add_argument("--output", type=Path, help="Output directory; default is paper source for complete studies.")
    parser.add_argument("--pilot-only-qa", action="store_true", help="Verified pilot preview outside manuscript only.")
    args = parser.parse_args()
    study = args.study
    if study is None:
        candidates = [parent / relative for parent in HERE.parents
                      for relative in ["followup_studies/work/rms_replication", "work/rms_replication"]]
        study = next((p for p in candidates if (p / "protocol.json").exists()), None)
    if study is None:
        parser.error("Supply --study /path/to/rms_replication.")
    output = args.output.resolve() if args.output else HERE.parent
    if args.pilot_only_qa:
        if args.output is None or output.is_relative_to(HERE.parent):
            parser.error("Pilot QA requires --output outside the manuscript source directory.")
    try:
        events, summary, sources = load_study(study.resolve(), args.pilot_only_qa)
        manifest = make_figure(events, output, args.pilot_only_qa, sources, summary)
    except (ValueError, KeyError, StopIteration) as exc:
        parser.error(str(exc))
    print(json.dumps({"mode": manifest["mode"], "events": len(events), "outputs": list(manifest["outputs"])}))


if __name__ == "__main__":
    main()

"""Generate manuscript tables only from a complete, verified RMS study.

This script reads results; it never trains, changes checkpoints, or fills missing outcomes.
"""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def number(x):
    return f"{x:,}".replace(",", "{,}") if x is not None else "--"


def pct(x):
    return f"{100*x:.2f}"


def sci(x):
    if not x:
        return "$0$"
    a, b = f"{x:.2e}".split("e")
    return f"${a}\\times10^{{{int(b)}}}$"


def table(columns, caption, label, header, rows):
    return "\n".join([
        r"\begin{table}[htbp]", r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        "\\caption{" + caption + "}", "\\label{" + label + "}",
        "\\begin{tabular}{" + columns + "}", r"\toprule",
        header + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def main(study, destination):
    verification = json.loads((study / "verification.json").read_text())
    summary = json.loads((study / "summary.json").read_text())
    assert verification["complete"] and summary["prospective_completed"] == 4
    assert not summary["remaining"]
    destination.mkdir(parents=True, exist_ok=True)
    events = sorted(summary["event_summaries"], key=lambda r: r["seed"])
    status_names = {
        "confirmed_grokking_with_failure": "Joint failure",
        "confirmed_grokking_event_free_completion": "Event-free completion",
        "failure_to_confirm_grokking": "No confirmation",
    }
    rows = [r"$0$ (pilot) & $19{,}900$ & $28{,}495$ & Joint failure \\"]
    for r in summary["runs"]:
        rows.append(f"${r['seed']}$ & ${number(r['confirmation_step'])}$ & ${number(r['terminal_step'])}$ & {status_names[r['status']]}" + r" \\")
    text = table("lrrl", "Accurate-CE RMS outcomes. Seed $0$ is the historical CPU pilot; seeds $1$--$4$ form the prospective CPU cohort. Prospective endpoints are first joint failure or step $100{,}000$. The pilot continued to $30{,}000$; its displayed endpoint is the separately analyzed first event.", "tab:rms-replication-outcomes", "Seed & Confirmation & Event / endpoint & Outcome", rows)
    (destination / "rms-outcomes.tex").write_text(text)

    rows = []
    for r in events:
        d = r["decoders"]["cv_selected"]
        rows.append(f"${r['seed']}$ & ${number(r['failed_step'])}$ & {pct(r['native_previous']['test']['accuracy'])} & {pct(r['native_failed']['test']['accuracy'])} & {pct(d['previous']['classification']['heldout']['accuracy'])} & {pct(d['failed']['classification']['heldout']['accuracy'])} & {'Yes' if d['previous']['refit_converged'] else 'No'} / {'Yes' if d['failed']['refit_converged'] else 'No'}" + r" \\")
    (destination / "rms-event-decoders.tex").write_text(table("lrrrrrl", "Adjacent RMS events and training-only decoders. Accuracies are percentages; before means one update before the displayed event. Decoder inputs include final RMS normalization. The last column gives selected-refit convergence before/after. Seed $0$ is the historical pilot; seeds $1$--$4$ are prospective.", "tab:rms-event-decoders", r"Seed & Event step & \multicolumn{2}{c}{Native test} & \multicolumn{2}{c}{Decoder test} & Converged \\ & & Before & After & Before & After &", rows))

    for part, start in enumerate(range(0, len(events), 3), 1):
        rows = []
        for r in events[start:start+3]:
            for swap in r["swaps"]:
                a, b = swap["metrics"]["train"], swap["metrics"]["test"]
                rows.append(f"${r['seed']}$ & {swap['mask']} & {pct(a['accuracy'])} & {pct(b['accuracy'])} & {a['accurate_ce64']:.5g} & {b['accurate_ce64']:.5g}" + r" \\")
            rows.append(r"\addlinespace")
        (destination / f"rms-swaps-{part}.tex").write_text(table("lrrrrr", "All eight parameter-group swaps at captured RMS failures. Seed $0$ is the historical pilot; seeds $1$--$4$ are prospective. Mask order is embeddings, hidden matrices, readout; $0$ retains the preceding state and $1$ uses the following state. Accuracy is in percent. CE is accurate float64 evaluation of the resulting float32 logits. Raw records additionally retain exact counts and float32 accurate CE.", f"tab:rms-swaps-{part}", "Seed & EHR & Train & Test & Train CE & Test CE", rows))

    rows = []
    for r in events:
        for state, d in r["derivatives"].items():
            rows.append(f"${r['seed']}$ & {'Before' if state == 'previous' else 'After'} & {sci(d['relative_l2_error'])} & {sci(d['absolute_l2_error'])} & {sci(d['reference_l2_norm'])} & {sci(d['class_sum_residual_l2'])}" + r" \\")
    (destination / "rms-derivatives.tex").write_text(table("llrrrr", "Accurate-CE derivative diagnostics on identical stored float32 training logits. Seed $0$ is the historical pilot; seeds $1$--$4$ are prospective. Errors compare the full mean-loss derivative with the analytic float64 reference. Class-sum residual is the $L_2$ norm across examples of the sum over classes. Maximum absolute errors and reference class-sum residuals are retained in the supporting records.", "tab:rms-derivatives", "Seed & State & Relative error & Absolute error & Reference norm & Class-sum residual", rows))

    rows = []
    for r in events:
        for state in ["previous", "failed"]:
            cv = r["decoders"]["cv_selected"][state]
            u = r["decoders"]["unregularized_control"][state]
            candidates = sum(x["converged"] for x in cv["candidate_convergence"])
            rows.append(f"${r['seed']}$ & {'Before' if state=='previous' else 'After'} & {sci(cv['selected_regularization'])} & {candidates}/5 & {pct(u['classification']['heldout']['accuracy'])} & {'Yes' if u['refit_converged'] else 'No'}" + r" \\")
    (destination / "rms-decoder-diagnostics.tex").write_text(table("llrrrl", "RMS decoder convergence and unregularized sensitivity. Seed $0$ is the historical pilot; seeds $1$--$4$ are prospective. Candidate convergence reports successful validation-candidate fits out of five; selected-refit convergence is in Table~\\ref{tab:rms-event-decoders}. Unregularized accuracy is held-out percent. Complete termination messages, objectives, iteration counts, and gradient norms accompany every fit.", "tab:rms-decoder-diagnostics", r"Seed & State & Selected $\lambda$ & Candidates & Unreg.\ test & Unreg.\ converged", rows))
    audit = {"complete_source_required": True, "summary_sha256": hashlib.sha256((study/"summary.json").read_bytes()).hexdigest(), "verification_sha256": hashlib.sha256((study/"verification.json").read_bytes()).hexdigest(), "prospective_failure_count": summary["prospective_failure_count"], "prospective_seed_count": summary["prospective_seed_count"], "captured_event_tables": len(events), "tables": sorted(p.name for p in destination.glob("rms-*.tex"))}
    (destination / "rms-table-generation.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT.parent/"rms_replication")
    parser.add_argument("--output", type=Path, default=ROOT/"source/tables")
    args = parser.parse_args()
    main(args.study, args.output)

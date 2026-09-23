"""Write a compact report from the verified long-horizon summaries."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    summary = json.loads((HERE/'summary.json').read_text())
    verification = json.loads((HERE/'verification.json').read_text())
    assert summary['status']=='completed' and verification['status']=='passed'
    assert json.loads((HERE/'serialized_resume_verification.json').read_text())['status']=='passed'
    records = summary['records']
    addition = [row for row in records if row['operation']=='addition']
    subtraction = [row for row in records if row['operation']=='subtraction']
    assert len(addition)==5 and len(subtraction)==1
    events = sum(row['first_joint_failure_step'] is not None for row in addition)
    extra_updates = sum(row['extension_updates'] for row in records)
    text = f'''# Corrected loss through the original 100,000-step budget

All six prespecified extensions completed step 100,000, adding {extra_updates:,} updates. The five corrected addition branches recorded {events}/5 joint train/test failures below 90% from step 6,000 through 100,000. The corrected subtraction branch recorded {'a joint event' if subtraction[0]['first_joint_failure_step'] is not None else 'no joint event'} after its step-10,500 grokking confirmation through step 100,000.

Each addition branch continues the accurate-cross-entropy intervention made at the matched step-6,000 state. Subtraction uses accurate cross-entropy from initialization. The extensions restore the saved step-30,000 model, all three optimizers, and Torch random state exactly. The horizon equals the original depth-1 budget.

| Condition | Monitored interval | Training states | Test evaluations | Minimum test (%) | Final test (%) | First joint event |
|---|---:|---:|---:|---:|---:|---:|
'''
    for row in records:
        label = f'{row["operation"].capitalize()}, seed {row["seed"]}'
        event = str(row['first_joint_failure_step']) if row['first_joint_failure_step'] is not None else 'none'
        text += f'| {label} | {row["monitoring_start_step"]:,}–{row["end_step"]:,} | {row["monitored_train_states"]:,} | {row["test_evaluations"]:,} | {100*row["minimum_measured_test_accuracy"]:.6f} | {100*row["final_test_accuracy"]:.6f} | {event} |\n'
    text += f'''
Training accuracy is checked at every state. Test accuracy is evaluated every 100 steps and whenever training accuracy is below 90%, so every joint event is detected. The interval counts include both endpoints and count the shared step-30,000 boundary once. Subtraction minima and counts begin at confirmation; pre-grokking evaluations are excluded from those entries.

The minimum training accuracy over every interval in the table is {100*min(row['minimum_train_accuracy'] for row in records):.6f}%. The original matched stock trajectories already have captured joint failures before step 30,000. This comparison uses their binary event status. Their event counts, durations, and historical minima are not compared with those of the uniformly monitored corrected branches.

The results establish persistence of the corrected branches through the original budget in these selected seeds. Brief test-only excursions between scheduled evaluations remain outside detection, and the finite horizon does not establish indefinite stability.

## Verification

All {verification['passed']} completed-trajectory checks passed. These cover every training row, the exact held-out evaluation schedule, failure counts and minima, source hashes, seed and operation identity, exact restored states, the checkpoint grid, and direct replay of final and minimum-test checkpoints. The final checkpoint also equals the corresponding step-100,000 grid state, including optimizer and random states.

Before launch, all six sources passed bitwise comparisons between twelve uninterrupted source updates, the monitored runner, and a six-plus-six continuation restored from a captured snapshot. Separate checks at step 90,000 wrote and reloaded the midpoint checkpoint from disk; those six-plus-six continuations also matched uninterrupted updates bitwise in every condition. A deliberately nonfinite test state was saved and failed explicitly without producing a completed summary. Portable source-path tests verified all six original hashes, and six imported model/data/optimizer modules match source commit `6d64a981af75f1300d9060109e81552d48a81360` byte for byte.

`protocol.json` preserves the design recorded before training. `summary.json` and `summary.csv` contain the derived results. `verification.json`, `runner_verification.json`, `serialized_resume_verification.json`, `failure_verification.json`, `portable_path_verification.json`, and `source_code_verification.json` contain the checks. `runs/` contains the raw every-state trajectories, logs, metadata, and retained states; `checkpoint_manifest.json` supplies the state hashes.
'''
    (HERE/'report.md').write_text(text)
    print(text)


if __name__ == '__main__': main()

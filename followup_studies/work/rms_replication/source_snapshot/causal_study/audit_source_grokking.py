"""Record sustained-grokking qualification of the five step-6000 sources."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records=[]
    for seed in range(5):
        base=WORK/(f"experiments/seed{seed}_original" if seed in (0,4) else f"basis_study/seed{seed}_baseline")
        source=base/"step_006000.pt"
        trajectory=base/"trajectory.csv"
        rows=list(csv.DictReader(trajectory.open()))
        by_step={int(r["step"]):r for r in rows if r.get("test_accuracy") and int(r["step"])%100==0 and int(r["step"])<=6000}
        first=None
        for step in sorted(by_step):
            window=[by_step.get(step+100*i) for i in range(6)]
            if all(r is not None and float(r["test_accuracy"])>=.95 for r in window):
                first=window;break
        if first is None:raise AssertionError(f"Seed {seed} has no completed sustained-grokking window by source6000")
        row6000=by_step[6000]
        record={"seed":seed,"source_checkpoint":str(source),"source_sha256":sha(source),
                "trajectory":str(trajectory),"trajectory_sha256":sha(trajectory),
                "grokking_window_start":int(first[0]["step"]),"grokking_confirmed_step":int(first[-1]["step"]),
                "six_evaluation_steps":[int(r["step"]) for r in first],
                "six_test_accuracies":[float(r["test_accuracy"]) for r in first],
                "source_train_accuracy":float(row6000["train_accuracy"]),"source_test_accuracy":float(row6000["test_accuracy"]),
                "qualified_before_source":int(first[-1]["step"])<=6000}
        records.append(record)
    result={"criterion":"Six consecutive 100-step-grid held-out accuracies >=0.95, all completed by step6000.",
            "source_step":6000,"all_five_qualified":all(r["qualified_before_source"] for r in records),"seeds":records}
    out=HERE/"source_grokking_manifest.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    lines=["# Step-6000 source qualification","",result["criterion"],"",
           "| Seed | First qualifying evaluation | Six-evaluation confirmation | Source train accuracy | Source test accuracy |",
           "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r['seed']} | {r['grokking_window_start']} | {r['grokking_confirmed_step']} | {100*r['source_train_accuracy']:.4f}% | {100*r['source_test_accuracy']:.4f}% |" for r in records]
    lines += ["","All five sources satisfy the sustained-grokking condition before the arithmetic intervention. Checkpoint and historical-trajectory SHA-256 values and all six qualifying evaluations are recorded in `source_grokking_manifest.json`.",
              "","This qualification concerns accuracy. Existing gradient error at the source means these are early post-grokking branches with inherited stock-arithmetic history.",""]
    (HERE/"source_grokking_manifest.md").write_text("\n".join(lines))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()

"""Wait for a fixed-horizon dense capture and run the established decoder probes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from run_collapse import make_model_optimizers, generate_modular_addition_data, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    scripts = Path(__file__).resolve().parent
    status_path = args.out.with_name(args.out.name + "_followup.json")
    print("Waiting at 30-second intervals for the fixed-horizon capture outcome.", flush=True)
    while not (args.capture / "event.json").exists():
        if (args.capture / "summary.json").exists():
            summary = json.loads((args.capture / "summary.json").read_text())
            status = {"status": "no_event_within_capture_horizon", "capture_summary": summary}
            status_path.write_text(json.dumps(status, indent=2))
            print(json.dumps(status), flush=True)
            return
        time.sleep(30)
    event = json.loads((args.capture / "event.json").read_text())
    print("Captured event: " + json.dumps(event), flush=True)
    candidates = []
    for filename in ("pre_event_train_healthy.pt", "pre_event_joint_healthy.pt", "last_joint_healthy.pt"):
        path = args.capture / filename
        if not path.exists():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        model, _ = make_model_optimizers(state["seed"], checkpoint=state)
        train_x, train_y, test_x, test_y = generate_modular_addition_data(seed=state["seed"])
        train_metrics = evaluate(model, train_x, train_y)
        test_metrics = evaluate(model, test_x, test_y)
        candidates.append({
            "path": str(path.resolve()), "step": state["step"],
            "train": train_metrics, "heldout": test_metrics,
            "qualifies": train_metrics["accuracy"] >= 0.99 and test_metrics["accuracy"] >= 0.99,
        })
    qualifying = [candidate for candidate in candidates if candidate["qualifies"]]
    if not qualifying:
        raise RuntimeError("No candidate has independently verified >=99% train and held-out accuracy")
    healthy = max(qualifying, key=lambda candidate: candidate["step"])
    selection = {
        "selection_status": "exploratory_second_seed_replication",
        "event": event,
        "healthy_rule": "Latest available designated pre-event healthy checkpoint independently verified at >=99% on both original train and held-out examples.",
        "candidates": candidates, "selected_healthy": healthy,
    }
    args.out.with_name(args.out.name + "_selection.json").write_text(json.dumps(selection, indent=2))
    print("Selected healthy checkpoint: " + json.dumps(healthy), flush=True)
    environment = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, str(scripts / "run_alignment.py"), "--healthy", healthy["path"],
               "--collapsed", str(args.capture / "collapse.pt"), "--out", str(args.out), "--decoder"]
    subprocess.run(command, env=environment, check=True)
    (args.out / "selection_protocol.json").write_text(json.dumps(selection, indent=2))
    command = [sys.executable, str(scripts / "run_whitened_probe.py"), "--features", str(args.out / "features.npz"),
               "--out", str(args.out / "decoder_whitened.json")]
    subprocess.run(command, env=environment, check=True)
    status = {"status": "completed", "event_step": event["step"], "healthy_step": healthy["step"], "results": str(args.out.resolve())}
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()

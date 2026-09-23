"""Independent read-only check of completed RMS measurements and fitted arrays."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
from scipy.special import logsumexp

B = Path(__file__).resolve().parent
R = B.parent / "rms_replication"


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b, rtol=1e-10, atol=1e-13):
    assert np.isclose(a, b, rtol=rtol, atol=atol), (a, b)


summary = read(R / "summary.json")
verification = read(R / "verification.json")
protocol = read(R / "protocol.json")
assert verification["complete"] and not summary["remaining"]
assert summary["prospective_completed"] == summary["prospective_seed_count"] == 4
assert verification["protocol_sha256"] == digest(R / "protocol.json")
for filename, wanted in protocol["immutable_hashes"].items():
    assert digest(R / filename) == wanted
for item in read(R / "source_manifest.json"):
    assert digest(R / "source_snapshot" / item["source"]) == item["sha256"]
events, monitoring = [], []
for run in summary["runs"]:
    seed = run["seed"]
    rows = list(csv.DictReader((R / f"runs/seed{seed}/trajectory.csv").open()))
    assert [int(r["step"]) for r in rows] == list(range(run["terminal_step"] + 1))
    streak, confirmed, first, failures = 0, None, None, []
    tested, after_confirm, below95, minimum = 0, 0, 0, None
    for row in rows:
        step = int(row["step"])
        train = int(row["train_correct"]) / int(row["train_total"])
        assert train == float(row["train_accuracy"])
        scheduled = step % 100 == 0
        test_expected = scheduled or (confirmed is not None and train < .9) or step == 100000
        assert bool(row["test_accuracy"]) == test_expected
        if test_expected:
            acc = int(row["test_correct"]) / int(row["test_total"])
            assert acc == float(row["test_accuracy"])
            tested += 1
        if scheduled:
            streak = streak + 1 if acc >= .95 else 0
            if streak >= 6 and confirmed is None:
                confirmed, first = step, step - 500
        joint = bool(confirmed is not None and step > confirmed and train < .9 and test_expected and acc < .9)
        assert (row["joint_failure"] == "True") == joint
        assert (row["grok_confirmed"] == "True") == (confirmed is not None)
        if joint:
            failures.append(step)
        if confirmed is not None and test_expected:
            after_confirm += 1
            below95 += acc < .95
            minimum = acc if minimum is None else min(minimum, acc)
    assert run["confirmation_step"] == confirmed and run["first_of_confirmation_streak"] == first
    assert run["test_evaluations"] == tested and run["post_confirmation_test_evaluations"] == after_confirm
    assert run["post_confirmation_below95_evaluations"] == below95
    assert run["minimum_post_confirmation_sampled_test_accuracy"] == minimum
    assert failures == [run["terminal_step"]] and run["status"] == "confirmed_grokking_with_failure"
    monitoring.append(dict(seed=seed, confirmation_step=confirmed, terminal_step=run["terminal_step"],
                           every_state_rows=len(rows), test_evaluations=tested,
                           post_confirmation_evaluations_inclusive=after_confirm,
                           below95_evaluations=below95, minimum_sampled_test_accuracy=minimum,
                           all_monitoring_recomputed=True))

for event in summary["event_summaries"]:
    folder = R / "analyses" / ("historical_seed0" if event["cohort"] == "historical_cpu_pilot" else f"prospective_seed{event['seed']}")
    assert read(folder / "summary.json") == event
    for item in read(folder / "artifact_manifest.json"):
        assert digest(folder / item["path"]) == item["sha256"]
    replay = read(folder / "replay_verification.json")
    assert replay["passed"] and replay["metrics_exact"] and replay["metrics"] == event["native_failed"]
    assert event["failed_step"] == event["previous_step"] + 1
    assert replay["from_step"] == event["previous_step"] and replay["to_step"] == event["failed_step"]
    analysis_protocol = read(folder / "analysis_protocol.json")
    features = np.load(folder / "decoder_features.npz")
    weights = np.load(folder / "decoder_weights.npz")
    reports = read(folder / "decoders.json")
    assert int(features["train"].sum()) == 3830 and int(features["heldout"].sum()) == 8939
    assert np.all(features["train"] ^ features["heldout"])
    expected = np.zeros(12769, dtype=bool)
    expected[np.random.default_rng(0).permutation(np.flatnonzero(features["train"]))[:766]] = True
    assert np.array_equal(features["validation"], expected)
    assert analysis_protocol["validation_indices"] == np.flatnonzero(expected).tolist()
    derivative_checks, decoder_checks = [], []
    for state, hkey in [("previous", "H0"), ("failed", "H1")]:
        arrays = np.load(folder / f"{state}_derivative_arrays.npz")
        z, y, g = arrays["logits"], arrays["targets"], arrays["accurate32_gradient"]
        assert z.dtype == g.dtype == np.float32 and arrays["reference64_gradient"].dtype == np.float64
        e = np.exp(z.astype(np.float64) - z.astype(np.float64).max(1, keepdims=True))
        den = e.sum(1, keepdims=True)
        oracle = e / den
        e[np.arange(len(y)), y] = 0
        oracle[np.arange(len(y)), y] = -e.sum(1) / den[:, 0]
        oracle /= len(y)
        np.testing.assert_allclose(oracle, arrays["reference64_gradient"], rtol=1e-13, atol=1e-30)
        delta = g.astype(np.float64) - oracle
        measured = dict(absolute_l2_error=float(np.linalg.norm(delta)), relative_l2_error=float(np.linalg.norm(delta) / np.linalg.norm(oracle)),
                        reference_l2_norm=float(np.linalg.norm(oracle)), maximum_absolute_error=float(abs(delta).max()),
                        class_sum_residual_l2=float(np.linalg.norm(g.astype(np.float64).sum(1))))
        for name, value in measured.items():
            close(value, event["derivatives"][state][name], rtol=1e-7, atol=1e-18)
        derivative_checks.append(dict(checkpoint=state, **measured))
        h = features[hkey].astype(np.float64)
        # Refit preconditioner is independently rebuilt using original training rows only.
        _, singular, vt = np.linalg.svd(h[features["train"]], full_matrices=False)
        independent_transform = (vt.T * (np.sqrt(3830) / np.maximum(singular, singular[0] * 1e-6))) @ vt
        for method in ["cv_selected", "unregularized_control"]:
            report = reports[method][state]
            transform = weights[f"{state}_{method}_feature_matrix"]
            w = weights[f"{state}_{method}_readout"]
            np.testing.assert_allclose(transform, independent_transform, rtol=1e-9, atol=1e-8)
            np.testing.assert_allclose(w, transform @ weights[f"{state}_{method}_scaled_readout"], rtol=1e-11, atol=1e-9)
            assert report["initialization"] == "zero" and w.dtype == np.float64
            assert report["split"]["inner_validation_indices"] == np.flatnonzero(expected).tolist()
            assert report["feature_transform"]["centered"] is False
            assert report["feature_transform"]["dimension"] == 128
            best = min(report["candidates"], key=lambda r: (r["validation"]["cross_entropy"], r["regularization"]))
            assert best["regularization"] == report["selected_regularization"]
            predictions = {}
            for name, mask in [("train", features["train"]), ("heldout", features["heldout"])]:
                scores, labels = h[mask] @ w, features["y"][mask]
                correct = int((scores.argmax(1) == labels).sum())
                ce = float(logsumexp(scores - scores[np.arange(len(labels)), labels, None], axis=1).mean())
                assert correct == report["classification"][name]["correct_count"]
                close(ce, report["classification"][name]["cross_entropy"])
                predictions[name] = dict(correct=correct, total=len(labels), accuracy=correct / len(labels), cross_entropy=ce)
            assert report["refit"]["iteration_limit"] == 1000
            assert all(c["optimization"]["iteration_limit"] == 500 for c in report["candidates"])
            decoder_checks.append(dict(checkpoint=state, method=method, predictions=predictions,
                                       refit_converged=report["refit"]["converged"], refit_iterations=report["refit"]["iterations"],
                                       selected_regularization=report["selected_regularization"],
                                       training_only_transform_recomputed=True))
    assert [s["mask"] for s in event["swaps"]] == [f"{x:03b}" for x in range(8)]
    assert event["swaps"][0]["metrics"] == event["native_previous"] and event["swaps"][-1]["metrics"] == event["native_failed"]
    for swap in event["swaps"]:
        assert swap["group_order"] == ["embeddings", "hidden", "readout"]
        for m in swap["metrics"].values():
            assert m["accuracy"] == m["correct"] / m["total"]
    events.append(dict(seed=event["seed"], cohort=event["cohort"], previous_step=event["previous_step"], failed_step=event["failed_step"],
                       replay_verified_keys=replay["exact_state_keys"], derivative_checks=derivative_checks,
                       decoder_checks=decoder_checks, swaps_heldout_percent={s["mask"]: 100 * s["metrics"]["test"]["accuracy"] for s in event["swaps"]}))

result = dict(status="passed", audit_utc=datetime.now(timezone.utc).isoformat(),
              all_prospective_runs_complete=True, prospective_joint_failures=4, prospective_seeds=4,
              historical_pilot_separate=True, event_count=len(events),
              model_updates_or_probe_fits_performed=False, study_files_modified=False,
              monitoring=monitoring, events=events,
              derivative_pairs_recomputed=10, decoder_prediction_splits_recomputed=40,
              full_dimension_training_only_SVDs_recomputed=10,
              scope="Replay equality is inspected from full-state verifier code and saved verification; no optimizer update was rerun in this audit. Low bounded-decoder accuracy leaves stronger decoders unresolved.",
              source_sha256={x: digest(R / x) for x in ["summary.json", "verification.json", "protocol.json", "analyze_event.py", "verify_and_summarize.py"]})
(B / "rms_completed_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["status", "prospective_joint_failures", "event_count", "derivative_pairs_recomputed", "decoder_prediction_splits_recomputed"]}))

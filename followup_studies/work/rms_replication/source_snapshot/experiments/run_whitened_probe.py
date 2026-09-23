"""Run declared decoder conditioning controls without replacing primary reports."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from decoder_probe import DEFAULT_REGULARIZATION_GRID, fit_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--refit-max-iter", type=int, default=1000)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to replace {args.out}")
    features = np.load(args.features)
    protocol = {
        "purpose": "Optimization conditioning sensitivity control; exploratory, not a replacement primary analysis.",
        "feature_source": str(args.features.resolve()),
        "feature_transform": "uncentered full-dimensional SVD whitening",
        "relative_singular_floor": 1e-6,
        "native_feature_dtype": "float32",
        "initialization": "zero",
        "regularization_grid": list(DEFAULT_REGULARIZATION_GRID),
        "selection": "original-training inner-validation cross-entropy; test rows untouched",
        "validation_seed": 0,
        "max_iter": args.max_iter,
        "refit_max_iter": args.refit_max_iter,
        "extra_control": "Unregularized whitened fit to separate preconditioning from changes in the L2 prior; reuse the CV fit if its selected lambda is zero.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_name(args.out.stem + "_protocol.json").write_text(json.dumps(protocol, indent=2))
    start = time.monotonic()
    results = {"protocol": protocol, "cv_selected": {}, "unregularized_control": {}}
    arrays = {}
    for name, key in (("healthy", "H0"), ("collapsed", "H1")):
        options = dict(
            n_classes=features["W0"].shape[1],
            feature_transform="whiten", whitening_relative_floor=1e-6,
            native_feature_dtype="float32", max_iter=args.max_iter,
            refit_max_iter=args.refit_max_iter,
        )
        result = fit_decoder(features[key], features["y"], features["train"], features["heldout"], **options)
        results["cv_selected"][name] = result.report
        if result.report["selected_regularization"] == 0.0:
            unregularized = result
        else:
            unregularized = fit_decoder(
                features[key], features["y"], features["train"], features["heldout"],
                regularization_grid=(0.0,), **options,
            )
        results["unregularized_control"][name] = unregularized.report
        for kind, fit in (("cv", result), ("unregularized", unregularized)):
            arrays[f"{name}_{kind}_readout"] = fit.readout
            arrays[f"{name}_{kind}_transformed_readout"] = fit.scaled_readout
            arrays[f"{name}_{kind}_feature_matrix"] = fit.feature_matrix
        print(json.dumps({
            "condition": name,
            "cv_regularization": result.report["selected_regularization"],
            "cv_converged": result.report["refit"]["converged"],
            "cv_iterations": result.report["refit"]["iterations"],
            "cv_heldout": result.report["classification"]["heldout"],
            "unregularized_converged": unregularized.report["refit"]["converged"],
            "unregularized_iterations": unregularized.report["refit"]["iterations"],
            "unregularized_heldout": unregularized.report["classification"]["heldout"],
        }), flush=True)
    results["elapsed_seconds"] = time.monotonic() - start
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False))
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()

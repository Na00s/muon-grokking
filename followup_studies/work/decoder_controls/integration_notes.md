# Integration notes

The matched decoder reaches 0.04–0.23% held-out accuracy at initialization and 0.02–1.12% at sustained memorization, compared with 99.40–100% at the existing references and 98.20–100% at the existing failed states. These early controls support the interpretation that the failed states retain task information accessible to this training-only decoder after training. The comparisons do not locate its precise emergence or identify a complete arithmetic algorithm.

All 20 CV-selected refits in the four-stage table converge. All five new memorization unregularized refits converge. Each initialization unregularized refit reaches its 1,000-iteration budget; held-out accuracies range from 0.18% to 0.62%. A claim of mathematically absent task information at initialization would exceed this evidence. All candidate statuses, objectives and gradient norms remain in the JSON reports.

The selected memorization steps are 200, 400, 200, 100 and 200. Their corresponding fifth training checks are 600, 800, 600, 500 and 600. The original five-check grokking starts for these CPU follow-up trajectories are 5,500, 5,500, 5,500, 5,300 and 5,400; these are separate provenance from the historical speed sweep.

Five initialization replays of 1,000 updates each exactly reproduce the recorded model tensors, optimizer states and CPU RNG, with all ten scheduled evaluation rows matching per seed. The selected checkpoint identity and validation indices were fixed before fitting. The phase-selection amendment was made before fitting and preserves the initial registration; seed 0 uses the same selected step under both rules. Verification replays add 5,000 updates and are excluded from experimental continuation counts.

Files: protocol.json, protocol_amendment.json, execution_commands.md, source_manifest.json, summary.json, summary.csv, report.md, stage_summary.json, verification.json, decoder_phase_controls_table.tex. Each seed folder includes early checkpoints, frozen feature arrays, CV and unregularized readouts/transforms, complete diagnostics, and replay evidence. No old result or manuscript source was edited.

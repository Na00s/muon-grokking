# Manuscript revision and controls

Status: all requested experiments and scientific integration are complete. Public distribution checks are recorded in `paper/verification.json` and `paper/audit/public_distribution/controls_20260923.json`.

## Manuscript organization

- Preserved the title and the abstract's central framing. Specified that the five readout-localized failures are unnormalized.
- Reordered the introduction around adjacent swaps, training-only decoders, the measured mean-dependent head displacement, and matched arithmetic controls. Explained the contribution relative to Liu et al. and Chou et al. near the beginning.
- Promoted the previous mechanism Figure 3 to opening Figure 1. Moved the spectral figure and its 0.9899 similarity observation to the later spectral discussion. Retained selected-trajectory, reused-seed, decoder-snapshot, and monitoring qualifications.
- Moved the detailed Section 6 analysis into the appendix. Kept the orbit-average identity and memorizer control in the main text. Compressed the depth account while retaining horizons, censoring, the matched depth-4 freeze, and appendix analyses.
- Clarified final-head compatibility when reading intermediate layers. Labeled accuracy axes in percentages and retained original plotted data and vector geometry.
- Preserved the Discussion's closing thesis and updated its normalized scope from the completed replication.

## Verified evidence and interpretation

- Speed comparisons now distinguish the five-seed median ratio (8,300/5,400 = 1.54), fastest successful sweep entries (6,300/5,400 = 1.17), and success-conditional means (30,485.7/13,011.1 = 2.34). The main text states selection asymmetry, differing grids, and the four excluded unsuccessful AdamW configurations.
- The tenfold hidden-LR continuation remains 5/5 joint failures, compared with 5/5 original stock and 0/5 accurate-CE branches through step 100,000. Its conclusion is limited to this reduction. The per-update hidden decay factor also changes from 0.997 to 0.9997.
- Table 2's minima of 16.77% and 43.57% reproduce their source logs. The controls have distinct run identities. Their initial scalar metrics agree and diverge by step 44,010. Historical checkpoint and runtime records are insufficient for a bitwise reconstruction. A separately registered CPU test gives identical results across three control implementations over 2,000 updates, including all 201 shared metric rows and terminal model, optimizer, and RNG states.
- Initialization and memorization controls use the same five unnormalized trajectories and existing decoder procedure. Held-out decoder accuracy is 0.04–0.23% at initialization, 0.02–1.12% at memorization, 99.40–100% at the existing references, and 98.20–100% at the existing failed states. All 20 selected refits converge. Initialization unregularized refits reach their iteration limit; all memorization sensitivity fits converge.
- Five early trajectory replays exactly reproduce the original step-1,000 model, optimizer, and CPU RNG states. Validation indices are shared across checkpoints. Forty independent saved-coefficient prediction checks pass.
- A new feature-mean timeline uses 49 Muon and 39 AdamW archived checkpoints, with full-grid means and actual irregular sampling shown. Its historical cohort and 100-update event raster are distinguished from the fresh, densely captured cohort.
- The historical accurate-CE RMS pilot is reanalyzed separately. All four swaps using the updated embeddings give 0.88–0.94% test accuracy; all four retaining the preceding embeddings give 96.21–96.41%. The matched decoder gives 97.37% before and 1.58% after, with converged selected refits. Its unregularized failed-state fit reaches its iteration limit at 2.62%. This pilot does not establish surviving high linear accessibility after its embedding failure.

## Definitions, references, and reproducibility

- Expanded definitions for original grokking, follow-up confirmation, strict sampled stability, joint failure, Table 2's training-only criterion, detected versus captured events, minima, final accuracy, evaluation counts, and seed incidence.
- Assigned G, C, I, and S to the four historical continuations around steps 44,700–44,710; preserved each measurement's associated trajectory and sampling resolution.
- Verified Liu et al. and Chou et al. against their primary texts. Added Wortsman et al. (ICLR 2024) and Molybog et al. (2023 preprint) with claims tied to their actual findings. Retained Prieto et al. in the numerical-stability discussion.
- Checked chance accuracy as 1/113, approximately 0.885%.
- New records preserve protocols, source and checkpoint hashes, exact execution commands, raw measurements, complete optimization diagnostics, and replay checks. Anonymous distribution copies retain scientific tensors and numerical values while removing identifying metadata.

## Completed RMS replication

All four prospectively registered seeds confirmed grokking and failed jointly: seeds 1–4 at steps 15,095, 15,789, 14,271 and 15,136, respectively. The registered first-event rule stopped all runs, totaling 60,291 new training updates. Every event replay passed. All eight parameter-group swaps were evaluated per event, including the separately analyzed historical pilot.

Embedding-only replacement induces joint failure in all four prospective cases. Retaining the old embeddings with the new hidden matrices and readout yields 98.41–100% test accuracy. Pre-update accurate-CE derivative relative errors remain below 9 × 10⁻⁸. Failed-state decoder accuracy is 17.35%, 64.86%, 93.82% and 2.65% for seeds 1–4, compared with 98.15–100% before the updates; all selected refits converge. The manuscript therefore scopes uniformly high surviving decodability to the tested unnormalized failures and reports the RMS variability directly.

All 70 new RMS table rows, the 40 swap evaluations, derivative arrays, decoder predictions and the 90 checkpoint hashes passed independent checks. A new appendix figure displays every swap and paired decoder outcome. No experiment remains to run.

## Delivery checks

The final main-text budget, typography, anonymization, archive reconstruction, repository synchronization and live-download checks are recorded in the accompanying verification files.

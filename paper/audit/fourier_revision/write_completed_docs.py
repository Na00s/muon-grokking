"""Write the release and revision notes from completed, audited results."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REPO=ROOT/'work/muon-grokking'
PAPER=REPO/'paper'
SUPP=REPO/'followup_studies'
summary=json.loads((ROOT/'work/long_horizon/summary.json').read_text())
assert summary['status']=='completed'
records=summary['records']
assert len(records)==6 and all(r['end_step']==100000 and r['first_joint_failure_step'] is None for r in records)
assert sorted((r['operation'], r['seed']) for r in records)==[('addition', seed) for seed in range(5)]+[('subtraction', 0)]
assert all(r['extension_updates']==70000 and r['arithmetic_start_step']==(6000 if r['operation']=='addition' else 0) for r in records)
addition=[r for r in records if r['operation']=='addition']
subtraction=next(r for r in records if r['operation']=='subtraction')
assert all(r['final_train_accuracy']==r['final_test_accuracy']==1 for r in addition)
audit=json.loads((ROOT/'work/long_horizon/verification.json').read_text())
assert audit['status']=='passed' and audit['passed']==audit['total'] and all(c['passed'] for c in audit['checks'])

report=f'''# Fourier interpretation and original-horizon revision

The revision addresses the task structure supplied by the Fourier filter, promotes the acute mean-dependent update result, extends corrected-loss branches to the original horizon, and centers the abstract and introduction on the failure mechanism and surviving information. The original experiments and reported results are retained.

## Fourier controls and interpretation

The complete addition family with DC equals averaging over pairs `(a+t, b-t)` with the same answer. A linear readout commutes with the average. The subtraction identity uses `(a+t, b+t)`.

On all five actual addition splits at p=113, a lookup predictor that memorizes training answers and returns zero on unseen inputs scores 0.761–0.906% on the held-out split before filtering and 100% afterward. The raw correct counts are 79, 68, 79, 81, and 77 out of 8,939. The zero logits use smallest-index argmax. A training-count-normalized lookup scores 100% after retaining any one of the 56 diagonal conjugate pairs plus DC, verified in 280 cases. Sparse sufficiency and phase sensitivity therefore also require care in algorithm identification.

The ablation denominators are explicit: the lookup retains 100% training accuracy, gives 0% held-out accuracy, and gives 29.9945% over the full grid. This control does not reproduce the trained models' near-chance full-grid ablation results. Donor-restricted averages at ten existing endpoints add information about unseen-input logits while still using task-defined answer orbits.

The paper preserves the early 95.14% filtered result and reframes its interpretation. Sections 5 and 6 now discuss task-informed projections and interference. Appendix K gives the proof, controls, exact counts, sparse counterexample, perturbation results, and donor analysis. Figure 9 shows the memorizer and sparse-control results. The original six figure PDFs and numerical cells in the 17 original appendix table files are retained.

## Acute mean-dependent readout failure

Section 4.4 now contains the decomposition of the actual head-induced logit change into a training-mean term and a centered term. Main-text Table 3 reports every seed. The mean term alone nearly reproduces each failing test accuracy; the largest observed difference is approximately 0.324421 percentage points. The centered term retains 87.20–100% test accuracy and at least 99.09% training accuracy.

This is an evaluation-only intervention using the post-update representation and its training mean. It establishes local sufficiency of a class-dependent offset shared across examples. Accuracy agreement does not establish prediction-by-prediction equivalence or a unique historical mediator. The raw-feature decoders, adjacent swaps, and numerical interventions provide separate evidence.

## Corrected loss through the original horizon

All five corrected addition branches and the corrected subtraction run completed step 100,000. Each continued for 70,000 updates from its preserved step-30,000 state, totaling 420,000 new updates. None recorded a joint train/test failure below 90%. The five addition branches finish at 100% test accuracy; their minimum measured test accuracy over steps 6,000–100,000 is {100*min(r['minimum_measured_test_accuracy'] for r in addition):.3f}%. The subtraction run finishes at {100*subtraction['final_test_accuracy']:.3f}%; its post-confirmation minimum is {100*subtraction['minimum_measured_test_accuracy']:.3f}%.

Every training state was checked. Test evaluation occurred every 100 steps and whenever training accuracy fell below 90%, detecting every joint event. Final and minimum-accuracy states were replayed. The extension confirms suppression through the original depth-1 budget. Behavior after step 100,000 and brief test-only excursions between scheduled evaluations retain their stated finite-measurement limits.

## Positioning and prose

The abstract and contribution list lead with acute failure localization, surviving information, and the numerical mechanism's scope. Liu et al.'s numerical feature-inflation mechanism and normalization studies, Chou et al.'s representation/readout probe diagnostic, and Wang's Muon speed/stability results are credited explicitly. The added evidence concerns actual captured displacements, matched long-horizon interventions, and the acute failure site under RMS normalization.

The four requested wording changes are applied: nearly constant Muon updates at small gradient norms; stabilization of tested depth-1 runs by freezing embeddings and readout; 100% accuracy of the filtered representation over the full grid; and a concrete comparison of numerical error and the failing parameter group between architectures.

## Submission format

The scientific main text occupies nine pages, meeting the initial-submission limit in the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines). The reproducibility and required AI use statements appear separately and are excluded from that limit. The compiled `sec:main-end` marker measures the scientific main text independently of the references' starting page.

The layout adjustment moves the quiet-window figure to Appendix C and the Fourier-grid figure to Appendix G, retaining their original PDF bytes and results. Four redundant sentences added during revision were removed. The causal follow-up figure remains in the main text. All six original figure PDFs and the numerical cells of the 17 original appendix table files are preserved. The AI use statement describes assistance with experiments, verification, derivations, literature comparison, figures, and revision.

The included style, bibliography style, natbib, and fancyhdr files are byte-identical to their counterparts in the [official ICLR 2027 template](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip). Existing filenames are retained; `audit/fourier_revision/control_audit/official_template_verification.json` records the comparisons and hashes.

## Verification and reproduction

The Fourier implementation passes 1,552 checks. An independent implementation passes 5,897 checks, including explicit Fourier synthesis of all 600 secondary perturbations. The audit corrected exact-zero tie handling in one auxiliary relocation control and a rounding error in the report; the primary scientific results are unchanged. A published-path reproduction independently matches the synthetic results and figure pixels. The earlier 604-check audit remains available with its immutable input snapshot.

Long-horizon verification checks all 420,006 extension states, test schedules, original hashes, restored model/optimizer/PyTorch random state, checkpoint grids, endpoint metrics, and minimum-accuracy metrics. The initial restoration and 12-update continuation checks are bitwise exact in all six branches. The source figure scripts and final publication checks record their input and output hashes.

Reproduction code and raw results live in `followup_studies/work/fourier_control` and `followup_studies/work/long_horizon`. The new checkpoint release uses `FOURIER_HORIZON_ARTIFACTS.json`; earlier source checkpoints remain in `CAUSAL_ARTIFACTS.json`. `paper/audit/fourier_revision` contains the independent review, and `paper/verification.json` records final compilation, preservation, and visual checks.
'''
(HERE/'revision_addendum.md').write_text(report)

source_readme='''# Revised submission source

Compile `main.tex` with Tectonic or a standard LaTeX/BibTeX installation. Figures, tables, bibliography, and the conference style are included.

```sh
tectonic main.tex
```

The revision preserves the title, anonymous authors, section order, all six original figure PDFs, and the numerical cells of the 17 original appendix table files. Section 4.4 contains the mean-dependent update equation and Table 3. The causal follow-up figure reports the swaps, decoders, and corrected-loss outcomes through step 100,000. Appendix J documents these experiments; Appendix K gives the Fourier averaging proof and memorizer controls, including Figure 9.

The scientific main text occupies nine pages. The reproducibility and required AI use statements are separate and excluded from the [ICLR 2027 initial-submission limit](https://iclr.cc/Conferences/2027/AuthorGuidelines). The quiet-window and Fourier-grid figures appear in Appendices C and G. Included template files are byte-identical to the official ICLR 2027 counterparts despite their retained legacy filenames.

The figure scripts use saved experimental results and record their input/output hashes. From a released `followup_studies` directory containing `work/`:

```sh
python figures/build_followup_figures.py --workspace /path/to/followup_studies --output .
python figures/build_fourier_control_figure.py --workspace /path/to/followup_studies --output .
```

The scripts resolve the data root automatically in the repository layout. In a standalone source archive, pass it explicitly. This source archive compiles without downloading experimental binaries.
'''
(PAPER/'source/README.md').write_text(source_readme)

p=PAPER/'audit_report.md';s=p.read_text()
if '## Latest Fourier and horizon revision' not in s:
    s=s.replace('# Audit of the revised submission\n','# Audit of the revised submission\n\n## Latest Fourier and horizon revision\n\nThe completed update adds Fourier memorizer controls, 420,000 training updates through the original 100,000-step horizon, a main-text mean-update table, and the four requested wording changes. See [revision_addendum.md](revision_addendum.md) for the results and [audit/fourier_revision/](audit/fourier_revision/) for the independent checks. The sections below record the preceding audit of the decoder, basis, and causal follow-ups.\n')
p.write_text(s)
p=PAPER/'revision_notes.md';s=p.read_text()
if '## Fourier controls and extended horizons' not in s:
    s=s.replace('# Selective revision and audit\n','# Selective revision and audit\n\n## Fourier controls and extended horizons\n\nThe latest changes are documented in [revision_addendum.md](revision_addendum.md). They add Appendix K and Figure 9, move the mean-update mechanism into Section 4.4 with Table 3, extend corrected-loss evidence to step 100,000, simplify the abstract/contribution list, and apply the four requested wording changes. The original numerical results are retained. `fourier_revision.patch` compares against commit ed5fa274735994a14d047ec73898b28d5ea1ed9f.\n\n## Prior revision and audit\n')
p.write_text(s)

p=REPO/'README.md';s=p.read_text()
start=s.index('The selectively revised manuscript') if 'The selectively revised manuscript' in s else s.index('The revised manuscript')
end=s.index('- [Revised paper]',start)
s=s[:start]+'''The revised manuscript centers on acute readout failures and surviving information. It adds Fourier memorizer controls and continues all five corrected addition branches and the subtraction control through the original 100,000-step horizon, with no joint train/test failure. The scientific main text is nine pages, with the exempt reproducibility and required AI use statements separate. All six original figure PDFs and numerical cells in the 17 original appendix tables are preserved; the quiet-window and Fourier-grid figures appear in Appendices C and G. The mean-dependent update is now in main-text Table 3; Appendix K and Figure 9 establish the task structure supplied by Fourier filtering.

'''+s[end:]
if '- [Fourier controls and 100,000-step results]' not in s:
    s=s.replace('- [New-experiment audit and claim review](paper/audit_report.md)','- [Fourier controls and 100,000-step results](paper/revision_addendum.md)\n- [New-experiment audit and claim review](paper/audit_report.md)')
s=s.replace('what happens to a grokked circuit after it forms','what happens after a model groks')
s=s.replace('once the circuit has formed','after sustained generalization')
anchor='## Follow-up experiments, September 22, 2026\n'
addition_note='''
The [Fourier controls](followup_studies/work/fourier_control/report.md) show that complete-family filtering is answer-orbit averaging: a lookup memorizer reaches 100% filtered accuracy on every seed. A training-count-normalized lookup also succeeds with any single diagonal frequency pair. The [long-horizon study](followup_studies/work/long_horizon/README.md) adds 420,000 updates and verifies all six continuations through step 100,000. Reproduction data are available in the [Fourier and horizon release](https://github.com/Na00s/muon-grokking/releases/tag/fourier-horizon-study-2026-09-22).

'''
if addition_note not in s:
    assert anchor in s
    s=s.replace(anchor,anchor+addition_note)
p.write_text(s)
p=SUPP/'README.md';s=p.read_text()
note='''**Latest update:** [Fourier controls](work/fourier_control/report.md) establish the task structure supplied by the filter. [Corrected-loss extensions](work/long_horizon/README.md) complete the original 100,000-step horizon in five addition branches and one subtraction run. The manuscript and [revision record](../paper/revision_addendum.md) incorporate these results. Download their saved checkpoints with `python followup_studies/download_artifacts.py --manifest FOURIER_HORIZON_ARTIFACTS.json` from the repository root.\n\n'''
if note not in s:
    s=s.replace('# Basis-change study: evidence and reproduction\n\n','# Basis-change study: evidence and reproduction\n\n'+note)
p.write_text(s)

release='''# Fourier controls and corrected-loss runs through 100,000 steps

This release contains every saved binary from the new Fourier-control and long-horizon studies, including restored states, the full checkpoint grid, endpoints, and resume-validation states. Code, protocols, complete CSV trajectories, numerical results, and independent audit records are committed with the revised paper.

The five corrected addition branches and the corrected subtraction control each continue from step 30,000 to step 100,000, adding 420,000 updates. All complete without a joint train/test failure. Fourier controls show that full-family and even sparse filtered sufficiency can arise from an externally supplied answer-orbit relation.

Use `followup_studies/FOURIER_HORIZON_ARTIFACTS.json` with the existing verified downloader. The earlier source checkpoints are in the causal-study release. Manifest hashes cover the archive and every saved binary. The manuscript source ZIP compiles independently of these binaries.
'''
(ROOT/'work/publish').mkdir(parents=True,exist_ok=True)
(ROOT/'work/publish/fourier_horizon_release_notes.md').write_text(release)
print('Wrote completed revision and release documentation.')

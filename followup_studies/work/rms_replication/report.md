# Accurate-CE RMS replication and localization

The prospective cohort contains seeds 1–4 on the same CPU backend and float32 update implementation as the historical seed 0 pilot. The pilot remains a separately identified observation. Runs stop at the first joint train/test failure after six scheduled test evaluations at or above 95%, or at 100,000 updates.

Prospective outcome: 4/4 seeds captured a joint failure. All recorded outcomes and event analyses passed the completion audit.

| Cohort | Seed | Grokking confirmation | First captured joint failure / endpoint | Outcome |
| --- | ---: | ---: | ---: | --- |
| Historical CPU pilot |0|19,900|28,495|Confirmed grokking with captured failure|
|Prospective CPU|1|11500|15095|confirmed_grokking_with_failure|
|Prospective CPU|2|12600|15789|confirmed_grokking_with_failure|
|Prospective CPU|3|10000|14271|confirmed_grokking_with_failure|
|Prospective CPU|4|12600|15136|confirmed_grokking_with_failure|

Historical seed 0 completed its original 30,000-update horizon, so its later states remain available. This replication analyzes its first captured failure using the same event procedures as the prospective cohort. Original every-update monitoring and event checkpoints identify that transition; its historical CSV records ordinary states every 10 updates. Prospective training counts are logged at every state.

## Adjacent update and fresh decoder results

| Cohort | Seed | Steps | Native test before | Native test after | Fresh decoder before | Fresh decoder after | CV refit convergence before/after |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
|historical_cpu_pilot|0|28494→28495|96.2076%|0.8838%|97.3711%|1.5774%|True/True|
|prospective_cpu|1|15094→15095|99.6980%|1.6333%|100.0000%|17.3509%|True/True|
|prospective_cpu|2|15788→15789|100.0000%|54.5475%|100.0000%|64.8618%|True/True|
|prospective_cpu|3|14270→14271|100.0000%|62.8594%|100.0000%|93.8248%|True/True|
|prospective_cpu|4|15135→15136|97.8745%|1.0292%|98.1542%|2.6513%|True/True|

Decoder inputs are the actual unembedding inputs after final RMS. Each transform and coefficient fit uses training examples only, with fixed 766-example internal validation subsets and validation-cross-entropy selection. The reports retain every candidate and refit status. Low accuracy from these bounded fits does not establish that no linear decoder exists.

## All parameter-group swaps

Mask order is embeddings, hidden matrices, readout; 1 uses the new state. Accuracies use float32 model evaluation.

| Cohort | Seed | Mask | Train | Test |
| --- | ---: | --- | ---: | ---: |
|historical_cpu_pilot|0|000|95.8747%|96.2076%|
|historical_cpu_pilot|0|001|95.8747%|96.2076%|
|historical_cpu_pilot|0|010|96.5013%|96.4090%|
|historical_cpu_pilot|0|011|96.5535%|96.4090%|
|historical_cpu_pilot|0|100|1.1488%|0.9397%|
|historical_cpu_pilot|0|101|1.1227%|0.9285%|
|historical_cpu_pilot|0|110|1.2010%|0.8950%|
|historical_cpu_pilot|0|111|1.1749%|0.8838%|
|prospective_cpu|1|000|99.5300%|99.6980%|
|prospective_cpu|1|001|99.5300%|99.7091%|
|prospective_cpu|1|010|100.0000%|100.0000%|
|prospective_cpu|1|011|100.0000%|100.0000%|
|prospective_cpu|1|100|1.0966%|1.6109%|
|prospective_cpu|1|101|1.0966%|1.6109%|
|prospective_cpu|1|110|1.0705%|1.6333%|
|prospective_cpu|1|111|1.0705%|1.6333%|
|prospective_cpu|2|000|100.0000%|100.0000%|
|prospective_cpu|2|001|100.0000%|100.0000%|
|prospective_cpu|2|010|100.0000%|100.0000%|
|prospective_cpu|2|011|100.0000%|100.0000%|
|prospective_cpu|2|100|54.5692%|53.7308%|
|prospective_cpu|2|101|54.5692%|53.7308%|
|prospective_cpu|2|110|55.2219%|54.5475%|
|prospective_cpu|2|111|55.2219%|54.5475%|
|prospective_cpu|3|000|100.0000%|100.0000%|
|prospective_cpu|3|001|100.0000%|100.0000%|
|prospective_cpu|3|010|100.0000%|100.0000%|
|prospective_cpu|3|011|100.0000%|100.0000%|
|prospective_cpu|3|100|58.9556%|57.9259%|
|prospective_cpu|3|101|58.9556%|57.9259%|
|prospective_cpu|3|110|64.1514%|62.8594%|
|prospective_cpu|3|111|64.1514%|62.8594%|
|prospective_cpu|4|000|97.6501%|97.8745%|
|prospective_cpu|4|001|97.6501%|97.8745%|
|prospective_cpu|4|010|99.3734%|98.3555%|
|prospective_cpu|4|011|99.3995%|98.4115%|
|prospective_cpu|4|100|0.8355%|1.0516%|
|prospective_cpu|4|101|0.8355%|1.0516%|
|prospective_cpu|4|110|0.8877%|1.0292%|
|prospective_cpu|4|111|0.8877%|1.0292%|

## Accurate derivative diagnostics

Each comparison uses identical stored float32 logits and an analytic float64 reference. Errors below summarize the complete mean-CE logit derivative.

| Cohort | Seed | State | Relative L2 error | Absolute L2 error | Reference norm | Class-sum L2 residual |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
|historical_cpu_pilot|0|previous|9.47645e-08|4.05651e-10|0.00428063|4.30088e-10|
|historical_cpu_pilot|0|failed|5.08076e-08|8.83297e-10|0.0173851|1.5549e-09|
|prospective_cpu|1|previous|8.11784e-08|1.004e-10|0.00123678|1.17823e-10|
|prospective_cpu|1|failed|3.88465e-08|6.34681e-10|0.0163382|1.4458e-09|
|prospective_cpu|2|previous|4.75929e-08|1.21046e-12|2.54336e-05|3.66946e-13|
|prospective_cpu|2|failed|4.86296e-08|6.0221e-10|0.0123836|9.578e-10|
|prospective_cpu|3|previous|8.98936e-08|1.91503e-12|2.13033e-05|9.9715e-13|
|prospective_cpu|3|failed|5.0415e-08|5.97572e-10|0.0118531|8.80307e-10|
|prospective_cpu|4|previous|5.27516e-08|1.39509e-10|0.00264464|2.35944e-10|
|prospective_cpu|4|failed|4.258e-08|7.15844e-10|0.0168117|1.37493e-09|

## Scope

These interventions localize the captured transitions within each RMS trajectory. The swaps distinguish the contributions of changed embeddings, hidden matrices and readout, including their interactions. They do not identify the complete upstream reason that the optimizer produced those changes. The RMS findings qualify claims about both numerical prevention and retained linear accessibility; claims about the unnormalized readout failures retain their separate supporting evidence.

## Reproducibility

`protocol.json` was frozen before launching seeds 1–4. `source_manifest.json` records the exact source snapshots. `launch.json` records each execution command, environment overrides and process identity. `runner_verification.json` checks original source initialization, monitored/manual updates and the historical event replay. Event subdirectories contain full replay verification, derivative arrays, all 8 swaps, actual readout-input features, decoder weights and candidate/refit diagnostics. `verification.json` recomputes trajectory endpoints, scheduled/triggered evaluation coverage, derivative errors and decoder predictions from raw records.

Prospective minimum sampled test accuracy and below-95% evaluation counts include the grokking-confirmation evaluation and end at the event or horizon. Historical replay verifies model, optimizer and torch RNG equality. Prospective replay additionally verifies Python and NumPy RNG states.

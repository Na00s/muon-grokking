# Fourier projector control protocol

Written before generating results, 2026-09-22.

## Primary mathematical claim

For a real vector-valued table h on Z_p x Z_p, retaining every Fourier mode (k,k), including (0,0), equals averaging h(a+t,b-t) over t. The subtraction family (k,-k), including (0,0), equals averaging h(a+t,b+t). Verify both with independently implemented FFT filtering, orbit reductions, and explicit translated-grid sums. Verify commutation with an arbitrary affine readout. Removing DC subtracts the global mean from the corresponding orbit average.

## Memorizer counterexample

Use p=113 and train fraction 0.30 with the repository's generate_modular_addition_data function, seeds 0,1,2,3,4, separately for addition and subtraction. A lookup table outputs a one-hot correct class on each training input and zero on every held-out input. This table makes no arithmetic computation at prediction time. It only looks up a stored vector.

Report integer correct counts and denominators for training, held-out, and complete grids. Argmax ties choose the smallest class index, matching NumPy/PyTorch. Also report expected accuracy under uniformly random tie breaking. Report exact orbit coverage, minimum/maximum training donor counts, and the hypergeometric probability that an orbit is uncovered under fixed-size sampling. The full family and DC projection should predict every orbit with at least one training donor correctly. No outcome is used to choose a seed or split.

Controls: omit DC; retain the opposite task family; use a seeded random bijection of output labels; zero all training donors before averaging; average training donors only. The random bijection memorizer evaluates against its correspondingly permuted labels, demonstrating arbitrary labels assigned to complete answer orbits. Held-out-donor averaging uses the known operation to group examples and remains task structured.

## Existing-checkpoint donor analysis

Use both endpoints of the five previously selected adjacent failure pairs, with no checkpoint selection based on Fourier results. Sources are basis_study/adjacent_seed0, basis_study/seed1_adjacent_event, basis_study/seed2_adjacent_event, basis_study/seed3_adjacent_event, and basis_study/adjacent_seed4. Reuse saved raw final-residual matrices and native heads. Reconstruct input order with the repository's seeded split and verify labels/masks. Recompute logits in float64 from stored float32-origin activations and weights.

Compare raw predictions, complete orbit averages, training-only donor averages, and held-out-only donor averages. For held-out evaluation also remove the evaluated row from its held-out orbit average to avoid self contribution. The known operation determines orbit membership in every averaged condition. This diagnoses reliance on training examples as donors; high donor-restricted accuracy does not independently identify an internal arithmetic algorithm. Readout swaps and training-only probes remain separate evidence.

## Numerical checks and reporting

Use deterministic CPU NumPy/Torch with fixed RNG seeds. Verify projector idempotence, average/FFT agreement, affine-readout commutation, and model-logit versus feature-filter agreement using scale-aware float64 tolerances. Record source hashes, software versions, exact command, all results, and checks. Original experiments and their reported numbers are outside this control's audit scope and will remain unchanged.

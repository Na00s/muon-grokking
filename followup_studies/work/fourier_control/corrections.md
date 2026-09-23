# Corrections after independent audit

The independent audit identified exact-zero argmax ties in the full-lookup frequency-relocation condition. Frequency relocation preserves the answer-zero orbit mathematically, because every Fourier character equals one there. Inverse-FFT cancellation left tiny numerical residuals that could change the smallest-index argmax on an otherwise all-zero held-out logit vector.

The corrected implementation assigns the reconstructed answer-zero row to the original template row before evaluation. It checks that the resulting full-lookup difference is exactly zero, then applies the declared smallest-index tie convention. This is restoration of a known exact structural identity. The isolated-family metrics and all manuscript control means drawn from them are unchanged. All secondary CSVs, verification records, summary files, figure assets, and the source manifest were regenerated. The primary protocol and extension protocol remain unchanged to preserve their original declarations.

A separate transcription correction changes the raw seed-1 post-event donor-table accuracy from 9.10% to 9.09%; its saved numerator and denominator are 813/8,939.

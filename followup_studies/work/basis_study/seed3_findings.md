# Fresh seed 3: endpoint geometry and acute readout failure

Seed 3 was one of three fresh seeds specified before training. Its first captured joint collapse is step 16308; the lowest test accuracy in the following 200 updates occurs at 16309. The latest measured checkpoint passing the joint >=99% criterion is 15800. The immediately preceding 16307 checkpoint already has degraded generalization, at 89.54% test accuracy, so it is labeled a preceding reference.

## Adjacent transition, 16307 to 16308

| Counterfactual | Test accuracy |
|---|---:|
| Preceding features and preceding head | 89.54% |
| Later features and preceding head | 91.01% |
| Preceding features and later head | 50.80% |
| Actual later model | 53.60% |

The updated features with the preceding readout improve accuracy slightly. The changed readout on preceding features reproduces most of the acute loss. This is the same qualitative local pattern observed in seeds 0 and 4.

In the exact bilinear logit decomposition, the readout-change, feature-change, and interaction norms are respectively 1.045, 0.0712, and 0.0552 times the total class-centered logit-change norm. All pairwise cross terms are retained in the full report; these ratios are not additive causal shares.

The train-fitted linear-map counterfactual reaches 52.80% test accuracy and 90.40% exact-class agreement with the actual model. The identity predictor reaches 50.80% accuracy and 86.03% agreement. Class-centered prediction error falls from 7.848% to 4.770% after fitting the map. This establishes predictive value for the representation fit while leaving the acute importance of the readout update clear.

The backward train-fitted linear map has a 10.66% held-out centered reconstruction error and 1.742% raw error, with condition number 53.50. Its transported readout reaches 90.49%, close to the preceding reference's 89.54%. The all-example oracle backward raw error is 1.700%, and the median and maximum principal angles are 6.66 and 16.01 degrees. Thus an exact global representation-wide basis transformation is also insufficient for this observed adjacent transition.

## Longer references

| Pair from reference 15800 | Actual test accuracy | Backward centered error | Backward transported accuracy | Linear-map counterfactual accuracy |
|---|---:|---:|---:|---:|
| First event, 16308 | 53.60% | 45.73% | 92.58% | 58.09% |
| Following minimum, 16309 | 32.12% | 45.01% | 94.68% | 33.93% |

Across these longer intervals both raw feature changes and readout changes are substantial. The preceding 15800 head on 16308 features gives 35.28%, while the 16308 head on 15800 features gives 48.52%. These longer comparisons combine earlier coadaptation and the acute failure. The adjacent two-by-two comparison provides the more focused evidence about the final update.

Geometry outputs, fitted maps, and native rounding controls were written directly into `seed3_first_event/`, `seed3_adjacent_event/`, and `seed3_peak_event/`, preserving the existing feature arrays, metadata, and original alignment outputs. Separate exact bilinear results are in `bilinear/` under the corresponding pair names. Aggregate geometry and bilinear CSV/JSON summaries include these results.

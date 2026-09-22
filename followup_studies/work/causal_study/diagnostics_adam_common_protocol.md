# Common-class gradient error through saved Adam state

Declared before measurement. Evaluate step 15000 and the exact immediate-pre-event checkpoint in each of the five seeds.

Let g be the float32 head gradient produced by an accurate float64 derivative of the same frozen float32 logits, cast to float32 for the shared network backward. Decompose e = g_stock - g into a common-class component c (mean across output classes, repeated for every class) and a centered component d = e - c. Apply the saved head Adam map separately to g, g_stock, g+c, and g+d, with identical parameters, first and second moments, step counter, learning rate, and weight decay.

Measure the change in the Adam head displacement relative to g and its centered component. Use both a float64 evaluation of Adam's formula and the realized float32 PyTorch update. The float64 formula adds c exactly at its working precision; the realized float32 input can introduce a small additional centered rounding component, which is measured separately. A linear SGD map with the saved head learning rate provides a mathematical control: adding an exactly common-class gradient produces an exactly common-class update up to evaluation rounding.

Evaluate the modified heads on the preceding features and on the actual following stock-update features. Held-out data are used only for evaluation. These tests establish whether numerical class-common gradient error can enter discriminating head updates through the saved elementwise preconditioner. A local conversion can coexist with no immediate accuracy change. It is not, by itself, sufficient to establish long-horizon causal accumulation.

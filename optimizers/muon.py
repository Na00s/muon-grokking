import torch
from torch import Tensor
from torch.optim import Optimizer


@torch.no_grad()
def zeroth_power_newton_schulz(
    matrix: Tensor,
    steps: int = 5,
    epsilon: float = 1e-7,
) -> Tensor:
    """
    Approximately replace the singular values of a matrix with values near 1.

    If:
        matrix = U @ S @ V.T

    this approximately returns:
        U @ V.T
    """
    if matrix.ndim != 2:
        raise ValueError("Muon expects two-dimensional weight matrices")

    original_dtype = matrix.dtype

    # Use float32 because this experiment runs on Apple MPS.
    x = matrix.float()

    was_transposed = False

    # Newton-Schulz is cheaper when rows <= columns.
    if x.shape[0] > x.shape[1]:
        x = x.T
        was_transposed = True

    # Scale the matrix so its spectral norm is safely below 1.
    x = x / (torch.linalg.vector_norm(x) + epsilon)

    # Quintic Newton-Schulz coefficients used by Muon.
    coefficient_a = 3.4445
    coefficient_b = -4.7750
    coefficient_c = 2.0315

    for _ in range(steps):
        gram = x @ x.T  # (rows, rows)

        polynomial = (
            coefficient_b * gram
            + coefficient_c * (gram @ gram)
        )  # (rows, rows)

        x = coefficient_a * x + polynomial @ x  # (rows, columns)

    if was_transposed:
        x = x.T

    return x.to(original_dtype)


class Muon(Optimizer):
    """
    Single-device Muon optimizer for hidden two-dimensional weight matrices.
    """

    def __init__(
        self,
        parameters,
        learning_rate: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.05,
        newton_schulz_steps: int = 5,
        nesterov: bool = True,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if not 0 <= momentum < 1:
            raise ValueError("momentum must be between 0 and 1")

        if weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

        defaults = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "newton_schulz_steps": newton_schulz_steps,
            "nesterov": nesterov,
        }

        super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["learning_rate"]
            momentum_coefficient = group["momentum"]
            weight_decay = group["weight_decay"]
            newton_schulz_steps = group["newton_schulz_steps"]
            use_nesterov = group["nesterov"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                if parameter.ndim != 2:
                    raise ValueError(
                        "Muon should only receive two-dimensional hidden weights"
                    )

                gradient = parameter.grad

                state = self.state[parameter]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)

                momentum_buffer = state["momentum_buffer"]

                # m_t = beta * m_(t-1) + (1 - beta) * gradient
                momentum_buffer.mul_(momentum_coefficient)
                momentum_buffer.add_(
                    gradient,
                    alpha=1.0 - momentum_coefficient,
                )

                if use_nesterov:
                    update = gradient.lerp(
                        momentum_buffer,
                        momentum_coefficient,
                    )
                else:
                    update = momentum_buffer.clone()

                update = zeroth_power_newton_schulz(
                    update,
                    steps=newton_schulz_steps,
                )

                rows, columns = parameter.shape

                # Shape-dependent scaling used by Muon.
                update_scale = max(1.0, rows / columns) ** 0.5
                update.mul_(update_scale)

                # Decoupled weight decay.
                parameter.mul_(1.0 - learning_rate * weight_decay)

                parameter.add_(
                    update,
                    alpha=-learning_rate,
                )

        return loss


if __name__ == "__main__":
    parameter = torch.nn.Parameter(
        torch.randn(128, 128)
    )

    optimizer = Muon([parameter])

    loss = parameter.square().mean()
    loss.backward()
    optimizer.step()

    print("Muon optimizer test passed")
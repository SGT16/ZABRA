from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
import torch.nn.functional as F


ACTIVATION_IMPLEMENTATION_VERSION = "fp32-stable-squareplus-v1"


class _StableSquareplus(torch.autograd.Function):
    """Squareplus with cancellation-safe values, analytic gradients, and the zero-radius subgradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        root = torch.hypot(x, alpha)
        denominator = 0.5 * root - 0.5 * x.clamp_max(0.0)
        denominator = torch.where(denominator > 0, denominator, 1.0)
        negative = 0.25 * alpha * (alpha / denominator)
        positive = 0.5 * x + 0.5 * root
        ctx.save_for_backward(x, alpha, root)
        return torch.where(x < 0, negative, positive)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, alpha, root = ctx.saved_tensors
        safe_root = torch.where(root > 0, root, 1.0)
        ratio = x / safe_root
        alpha_ratio = alpha / safe_root
        negative_dx = 0.5 * alpha_ratio.square() / (1.0 - ratio.clamp_max(0.0))
        dx = torch.where(x < 0, negative_dx, 0.5 * (1.0 + ratio))
        dx = torch.where(root > 0, dx, 0.0)
        da = torch.where(root > 0, 0.5 * alpha_ratio, 0.5)
        return grad_output * dx, (grad_output * da).sum_to_size(alpha.shape)


def stable_squareplus(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Return Squareplus in the input dtype; sensitive arithmetic is FP32+."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        result = _StableSquareplus.apply(x.to(dtype=dtype), alpha.to(dtype=dtype))
    return result.to(dtype=x.dtype)


class PELU(nn.Module):
    r"""Two scalar/site parameters, initialized at a=b=1 and projected >=0.1.

    f(x)=(a/b)x for x>=0, a*expm1(x/b) otherwise.  This is Eq. (2)
    of Trottier et al., https://arxiv.org/html/1605.09332v4 (ICMLA 2017,
    DOI: 10.1109/ICMLA.2017.00038).  Projection follows Section III.
    The common optimizer/no-decay policy is retained for matched comparisons;
    this is not a reproduction of that paper's complete training recipe.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.b = nn.Parameter(torch.tensor(float(b)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            work_x, a, b = x.to(dtype), self.a.to(dtype), self.b.to(dtype)
            negative = a * torch.expm1(work_x.clamp_max(0.0) / b)
            result = torch.where(work_x >= 0, (a / b) * work_x, negative)
        return result.to(x.dtype)

    @torch.no_grad()
    def project_parameters_(self) -> None:
        self.a.clamp_(min=0.1)
        self.b.clamp_(min=0.1)


class Swish(nn.Module):
    """Parameter-free Swish control: f0(x) = x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class ResidualSafeReLU(nn.Module):
    """ReLU using the input-side sign mask for backward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0.0, x, 0.0)


def _channel_parameter(parameter: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar/channel-wise parameter over NCHW or NC tensors."""
    if parameter.numel() == 1:
        return parameter
    if x.ndim < 2 or x.shape[1] != parameter.numel():
        raise ValueError(
            f"Channel-wise activation expected {parameter.numel()} channels, "
            f"but received shape {tuple(x.shape)}."
        )
    return parameter.view(1, -1, *([1] * (x.ndim - 2)))


class AconC(nn.Module):
    r"""Channel-wise ACON-C from Ma et al. (CVPR 2021).

    f(x) = (p1-p2)x*sigmoid(beta*(p1-p2)x) + p2*x
    """

    def __init__(self, channels: int | None = None) -> None:
        super().__init__()
        size = int(channels) if channels else 1
        self.p1 = nn.Parameter(torch.ones(size))
        self.p2 = nn.Parameter(torch.zeros(size))
        self.beta = nn.Parameter(torch.ones(size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = _channel_parameter(self.p1, x)
        p2 = _channel_parameter(self.p2, x)
        beta = _channel_parameter(self.beta, x)
        difference = p1 - p2
        return difference * x * torch.sigmoid(beta * difference * x) + p2 * x


class QianGated(nn.Module):
    r"""Qian et al. (2018) gated mixture of LReLU and ELU.

    f(x) = sigmoid(omega*x)*LReLU(x)
           + (1-sigmoid(omega*x))*ELU(x)
    """

    def __init__(self, channels: int | None = None, omega: float = 1.0) -> None:
        super().__init__()
        size = int(channels) if channels else 1
        self.omega = nn.Parameter(torch.full((size,), float(omega)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        omega = _channel_parameter(self.omega, x)
        gate = torch.sigmoid(omega * x)
        lrelu = F.leaky_relu(x, negative_slope=0.01)
        elu = F.elu(x, alpha=1.0)
        return gate * lrelu + (1.0 - gate) * elu


def _inverse_sigmoid_range(value: float, low: float, high: float) -> torch.Tensor:
    if not low < high:
        raise ValueError(f"Invalid parameter range [{low}, {high}].")
    value = min(max(float(value), low + 1e-6), high - 1e-6)
    y = (value - low) / (high - low)
    return torch.tensor(math.log(y / (1.0 - y)), dtype=torch.float32)


class SquareplusBase(nn.Module):
    """Constrained, optionally learnable Squareplus width."""

    variant = "Squareplus"
    ablation_role = "squareplus"
    formula = "unspecified"

    def __init__(
        self,
        alpha: float = 0.25,
        *,
        learnable: bool = True,
        learn_alpha: bool | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 1.5,
    ) -> None:
        super().__init__()
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        raw_alpha = _inverse_sigmoid_range(alpha, self.alpha_min, self.alpha_max)
        alpha_is_learnable = learnable if learn_alpha is None else learn_alpha
        if alpha_is_learnable:
            self.raw_alpha = nn.Parameter(raw_alpha)
        else:
            self.register_buffer("raw_alpha", raw_alpha)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)

    @property
    def learning_mode(self) -> str:
        """Report which shape parameters require gradients."""
        alpha_is_learned = isinstance(self.raw_alpha, nn.Parameter)
        return "alpha" if alpha_is_learned else "none"


class FixedSquareplus(SquareplusBase):
    """Fixed Squareplus, (x + sqrt(x² + α²)) / 2."""

    variant = "Squareplus-fixed"
    ablation_role = "base_squareplus_fixed"
    formula = "0.5*(x+sqrt(x^2+alpha^2))"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return stable_squareplus(x, self.alpha)


class LearnedSquareplus(FixedSquareplus):
    """Squareplus with a bounded learned α and width b = α²."""

    variant = "Squareplus-learned"
    ablation_role = "base_squareplus_learned_alpha_b_equals_alpha_squared"


@torch.no_grad()
def project_activation_parameters_(model: nn.Module) -> None:
    """Apply explicit projected constraints after each optimizer update."""
    for layer in model.modules():
        if isinstance(layer, PELU):
            layer.project_parameters_()


class LogMeanActivation(nn.Module):
    """Stable log-mean-exp hybrid activation."""

    def __init__(self, left: Callable[[torch.Tensor], torch.Tensor], right: Callable[[torch.Tensor], torch.Tensor], t: float = 0.6):
        super().__init__()
        if not 0.0 < t < 1.0:
            raise ValueError("t must be strictly between 0 and 1")
        self.left = left
        self.right = right
        self.register_buffer("log_t", torch.tensor(math.log(t), dtype=torch.float32))
        self.register_buffer("log_one_minus_t", torch.tensor(math.log(1.0 - t), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.left(x) + self.log_t.to(dtype=x.dtype)
        b = self.right(x) + self.log_one_minus_t.to(dtype=x.dtype)
        return torch.logsumexp(torch.stack((a, b), dim=0), dim=0)


class PowLU(nn.Module):
    """Jiang et al. (2026), arXiv:2605.25704, scalar Eq. 1 with m=3.

    The CNN activation uses one tensor at each existing activation site.
    The two-projection LLM block from the paper is a separate architecture.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.to(dtype=dtype)
            positive = value > 0
            base = torch.where(positive, value, 1.0)
            gain = torch.pow(base, 3.0 / (torch.sqrt(base) + 1.0))
            factor = torch.where(positive, gain, value)
            result = (value * torch.sigmoid(value)) * factor
        return result.to(dtype=x.dtype)


class ZABRA(nn.Module):
    """Zero-anchored bounded rational adaptation with a zero-frozen omitted term."""

    def __init__(self, mode="both"):
        super().__init__()
        if mode not in ("both", "odd", "even"):
            raise ValueError(f"Unknown ZABRA component mode: {mode}")
        self.variant = {"both": "ZABRA", "odd": "ZABRA-Odd", "even": "ZABRA-Even"}[mode]
        self.a = nn.Parameter(torch.zeros(()), requires_grad=mode != "even")
        self.c = nn.Parameter(torch.zeros(()), requires_grad=mode != "odd")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        odd = z / (1.0 + z.square())
        correction = self.a.tanh() * odd + self.c.tanh() * odd.square()
        return (torch.relu(z) + correction).to(x.dtype)


def _activation_key(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def canonical_activation_name(name: str) -> str:
    key = _activation_key(name)
    aliases = {
        "powlu": "PowLU",
        "zabra": "ZABRA",
        "zabraodd": "ZABRA-Odd",
        "zabraeven": "ZABRA-Even",
        "squareplus": "Squareplus-fixed",
        "squareplusfixed": "Squareplus-fixed",
        "squarepluscontrol": "Squareplus-fixed",
        "squarepluslearned": "Squareplus-learned",
    }
    return aliases.get(key, name)


def is_squareplus_activation(name: str) -> bool:
    return canonical_activation_name(name) in {"Squareplus-fixed", "Squareplus-learned"}


def make_activation(
    name: str,
    *,
    channels: int | None = None,
    log_t: float = 0.6,
    squareplus_learnable: bool = True,
    squareplus_learn_alpha: bool | None = None,
    squareplus_alpha: float = 0.25,
    squareplus_alpha_min: float = 0.05,
    squareplus_alpha_max: float = 1.5,
) -> nn.Module:
    key = _activation_key(canonical_activation_name(name))
    factories: dict[str, Callable[[], nn.Module]] = {
        "relu": lambda: nn.ReLU(inplace=False),
        "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=False),
        "prelu": lambda: nn.PReLU(num_parameters=1),
        "preluscalar": lambda: nn.PReLU(num_parameters=1),
        "preluchannel": lambda: nn.PReLU(num_parameters=channels or 1),
        "elu": lambda: nn.ELU(inplace=False),
        "celu": lambda: nn.CELU(alpha=1.0, inplace=False),
        "pelu": PELU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "powlu": PowLU,
        "zabra": ZABRA,
        "zabraodd": lambda: ZABRA("odd"),
        "zabraeven": lambda: ZABRA("even"),
        "swish": Swish,
        "silu": nn.SiLU,
        "mish": lambda: nn.Mish(inplace=False),
        "softplus": nn.Softplus,
        "acon": lambda: AconC(channels),
        "aconc": lambda: AconC(channels),
        "qian": lambda: QianGated(channels),
        "qiangated": lambda: QianGated(channels),
        "squareplusfixed": lambda: FixedSquareplus(
            alpha=squareplus_alpha, learnable=False, learn_alpha=False,
            alpha_min=squareplus_alpha_min, alpha_max=squareplus_alpha_max,
        ),
        "squarepluslearned": lambda: LearnedSquareplus(
            alpha=squareplus_alpha, learnable=True, learn_alpha=True,
            alpha_min=squareplus_alpha_min, alpha_max=squareplus_alpha_max,
        ),
        "ltr": lambda: LogMeanActivation(torch.tanh, F.relu, t=log_t),
        "ltl": lambda: LogMeanActivation(torch.tanh, lambda x: F.leaky_relu(x, 0.01), t=log_t),
        "lil": lambda: LogMeanActivation(lambda x: x, lambda x: F.leaky_relu(x, 0.01), t=log_t),
    }
    if key not in factories:
        raise ValueError(f"Unknown activation '{name}'. Available keys: {', '.join(sorted(factories))}")
    return factories[key]()


def _sample_sd(values: torch.Tensor) -> float:
    return float(values.std(unbiased=True)) if values.numel() > 1 else 0.0


def _stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_mean": None, f"{prefix}_sd": None, f"{prefix}_min": None, f"{prefix}_max": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        f"{prefix}_mean": float(tensor.mean()),
        f"{prefix}_sd": _sample_sd(tensor),
        f"{prefix}_min": float(tensor.min()),
        f"{prefix}_max": float(tensor.max()),
    }


def squareplus_parameter_rows(model: nn.Module) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for layer_name, layer in model.named_modules():
        if not isinstance(layer, SquareplusBase):
            continue
        row: dict[str, float | str] = {
            "layer": layer_name,
            "variant": layer.variant,
            "ablation_role": layer.ablation_role,
            "formula": layer.formula,
            "learning_mode": layer.learning_mode,
            "alpha": float(layer.alpha.detach().cpu()),
            "alpha_parameterization": "bounded_sigmoid",
            "alpha_min": layer.alpha_min,
            "alpha_max": layer.alpha_max,
            "sharing_group": getattr(layer, "alpha_sharing_group", layer_name),
            "numerical_implementation": ACTIVATION_IMPLEMENTATION_VERSION,
        }
        if isinstance(layer, FixedSquareplus):
            row["squareplus_b"] = float(layer.alpha.detach().square().cpu())
        rows.append(row)
    return rows


def squareplus_parameter_stats(model: nn.Module) -> dict[str, float | int | str | None]:
    """Return layer statistics for the fixed and learned Squareplus controls."""
    rows = squareplus_parameter_rows(model)
    result: dict[str, float | int | str | None] = {
        "variant": rows[0]["variant"] if rows else None,
        "ablation_role": rows[0]["ablation_role"] if rows else None,
        "formula": rows[0]["formula"] if rows else None,
        "learning_mode": rows[0]["learning_mode"] if rows else None,
        "count": len(rows),
    }
    result.update(_stats([float(row["alpha"]) for row in rows], "alpha"))
    result.update(_stats([float(row["squareplus_b"]) for row in rows if "squareplus_b" in row], "squareplus_b"))
    squareplus_layers = [layer for layer in model.modules() if isinstance(layer, SquareplusBase)]
    near_lower = [
        float(layer.alpha.detach().cpu()) <= layer.alpha_min + 0.005
        for layer in squareplus_layers
    ]
    result["alpha_near_lower_fraction"] = float(sum(near_lower) / len(near_lower)) if near_lower else None
    result["alpha_unique_parameter_count"] = len({
        id(layer.raw_alpha) for layer in squareplus_layers if isinstance(layer.raw_alpha, nn.Parameter)
    })
    result["alpha_sharing"] = getattr(model, "activation_sharing", "site")
    result["numerical_implementation"] = ACTIVATION_IMPLEMENTATION_VERSION
    return result


def squareplus_parameter_snapshot(model: nn.Module) -> dict[str, float | None]:
    stats = squareplus_parameter_stats(model)
    return {"squareplus_alpha": stats["alpha_mean"]}


def adaptive_activation_parameter_rows(model: nn.Module) -> list[dict[str, float | int | str]]:
    """Layer summaries for trainable comparison activations."""
    rows: list[dict[str, float | int | str]] = []
    for layer_name, layer in model.named_modules():
        if isinstance(layer, ZABRA):
            rows.append({
                "layer": layer_name, "activation": layer.variant, "channels": 1,
                "a": layer.a.detach().item(), "c": layer.c.detach().item(),
                "odd_coefficient": layer.a.detach().tanh().item(),
                "even_coefficient": layer.c.detach().tanh().item(),
            })
        elif isinstance(layer, AconC):
            rows.append({
                "layer": layer_name, "activation": "ACON-C", "channels": layer.p1.numel(),
                "p1_mean": float(layer.p1.detach().mean().cpu()),
                "p2_mean": float(layer.p2.detach().mean().cpu()),
                "beta_mean": float(layer.beta.detach().mean().cpu()),
            })
        elif isinstance(layer, QianGated):
            rows.append({
                "layer": layer_name, "activation": "Qian-Gated", "channels": layer.omega.numel(),
                "omega_mean": float(layer.omega.detach().mean().cpu()),
            })
        elif isinstance(layer, PELU):
            rows.append({
                "layer": layer_name, "activation": "PELU", "channels": 1,
                "a": float(layer.a.detach().cpu()), "b": float(layer.b.detach().cpu()),
                "constraint": "projected_min_0.1", "initial_a": 1.0, "initial_b": 1.0,
            })
        elif isinstance(layer, nn.PReLU):
            rows.append({
                "layer": layer_name, "activation": "PReLU", "channels": layer.weight.numel(),
                "negative_slope_mean": float(layer.weight.detach().mean().cpu()),
            })
    return rows


def adaptive_activation_parameter_stats(model: nn.Module) -> dict[str, float | int | None]:
    acon_layers = [layer for layer in model.modules() if isinstance(layer, AconC)]
    qian_layers = [layer for layer in model.modules() if isinstance(layer, QianGated)]
    pelu_layers = [layer for layer in model.modules() if isinstance(layer, PELU)]
    prelu_layers = [layer for layer in model.modules() if isinstance(layer, nn.PReLU)]
    result: dict[str, float | int | None] = {
        "adaptive_layer_count": (len(acon_layers) + len(qian_layers) + len(pelu_layers) + len(prelu_layers)
                                 + sum(isinstance(layer, ZABRA) for layer in model.modules())),
    }
    for prefix, values in (
        ("acon_p1", [float(value) for layer in acon_layers for value in layer.p1.detach().cpu().flatten()]),
        ("acon_p2", [float(value) for layer in acon_layers for value in layer.p2.detach().cpu().flatten()]),
        ("acon_beta", [float(value) for layer in acon_layers for value in layer.beta.detach().cpu().flatten()]),
        ("qian_omega", [float(value) for layer in qian_layers for value in layer.omega.detach().cpu().flatten()]),
        ("pelu_a", [float(layer.a.detach().cpu()) for layer in pelu_layers]),
        ("pelu_b", [float(layer.b.detach().cpu()) for layer in pelu_layers]),
        ("prelu_slope", [float(value) for layer in prelu_layers for value in layer.weight.detach().cpu().flatten()]),
    ):
        result.update(_stats(values, prefix))
    return result

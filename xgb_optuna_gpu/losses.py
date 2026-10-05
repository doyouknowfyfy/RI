"""Loss-function specifications: MSE, (pseudo-)Huber and Tweedie."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

# Search-space entry: (kind, low, high, options). kind in {"int", "float"};
# options may contain {"log": True} or {"step": 0.05}.
SpaceEntry = tuple[str, float, float, dict]


@dataclass(frozen=True)
class LossSpec:
    """How one loss maps onto XGBoost params / metrics.

    ``loss_metric`` is the XGBoost metric that corresponds to the training loss. It is
    always placed *last* in ``eval_metric`` so XGBoost's early stopping (which monitors
    the last metric on the last eval set, here ``valid``) tracks the loss itself.
    """

    name: str
    objective: str
    loss_metric: str
    fixed_params: dict[str, Any] = field(default_factory=dict)
    extra_space: dict[str, SpaceEntry] = field(default_factory=dict)

    def loss_metric_name(self, params: dict[str, Any]) -> str:
        if self.name == "tweedie":
            # XGBoost reports this metric under exactly this name.
            return f"tweedie-nloglik@{params['tweedie_variance_power']}"
        return self.loss_metric

    def eval_metrics(self, params: dict[str, Any], base_metrics: Sequence[str]) -> list[str]:
        loss_metric = self.loss_metric_name(params)
        return [m for m in base_metrics if m != loss_metric] + [loss_metric]


LOSSES: dict[str, LossSpec] = {
    "mse": LossSpec(name="mse", objective="reg:squarederror", loss_metric="rmse"),
    # huber_slope (delta) is in target units: set it per dataset, e.g. a high quantile of
    # |y - median(y)|. The default 1.0 is only sensible for targets of order ~1.
    "huber": LossSpec(
        name="huber",
        objective="reg:pseudohubererror",
        loss_metric="mphe",
        fixed_params={"huber_slope": 1.0},
    ),
    "tweedie": LossSpec(
        name="tweedie",
        objective="reg:tweedie",
        loss_metric="tweedie-nloglik",
        # Variance power p in (1, 2): ~1 Poisson-like, ~2 Gamma-like. Discretised so the
        # metric name (tweedie-nloglik@p) stays short and stable.
        extra_space={"tweedie_variance_power": ("float", 1.05, 1.95, {"step": 0.05})},
    ),
}


def get_loss(name: str) -> LossSpec:
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"Unknown loss {name!r}; choose from {sorted(LOSSES)}") from None

"""Loss functions.

Both losses take the raw network output on the *link* scale and the target in the
*transformed* space produced by :class:`tsfusion.data.TargetScaler`, and expose
``inverse_link`` which maps the network output back to that transformed space.

* :class:`MSELoss`      - identity link; pair with target mode ``standard`` (or ``log1p_standard``).
* :class:`TweedieLoss`  - log link (output = log mu); pair with target mode ``mean_scale``.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .config import TrainConfig


class MSELoss(nn.Module):
    name = "mse"
    compatible_target_modes = ("standard", "log1p_standard", "none", "mean_scale")

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((output.float() - target.float()) ** 2)

    @staticmethod
    def inverse_link(output: torch.Tensor) -> torch.Tensor:
        return output


class TweedieLoss(nn.Module):
    """Negative Tweedie log-likelihood (up to terms that do not depend on mu), log link.

    With ``f = log(mu)``::

        p = 1      : exp(f) - y f                                    (Poisson)
        p = 2      : y exp(-f) + f                                   (Gamma)
        otherwise  : -y exp((1-p) f) / (1-p) + exp((2-p) f) / (2-p)

    Valid for ``1 <= p <= 3``. For every such ``p`` the minimiser is ``mu = E[y | x]``,
    so ``p`` only changes how residuals are weighted (variance ~ mu^p): larger ``p``
    down-weights errors on large targets. ``p`` cannot be learned by minimising this
    loss (the normalising constant depends on p) - tune it on the validation set.

    Note: 1 < p < 2 (compound Poisson-Gamma) puts mass at y = 0. For strictly positive
    targets, values of p in [~1.5, 2.5] are usually the interesting range.
    """
    name = "tweedie"
    compatible_target_modes = ("mean_scale", "none")

    def __init__(self, power: float = 1.5, max_log: float = 30.0):
        super().__init__()
        if not 1.0 <= power <= 3.0:
            raise ValueError("Tweedie power must be in [1, 3]")
        self.power = float(power)
        self.max_log = max_log

    def forward(self, log_mu: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f = log_mu.float().clamp(-self.max_log, self.max_log)
        y = target.float()
        p = self.power
        if math.isclose(p, 1.0):
            loss = torch.exp(f) - y * f
        elif math.isclose(p, 2.0):
            loss = y * torch.exp(-f) + f
        else:
            loss = -y * torch.exp((1.0 - p) * f) / (1.0 - p) + torch.exp((2.0 - p) * f) / (2.0 - p)
        return loss.mean()

    @staticmethod
    def inverse_link(output: torch.Tensor) -> torch.Tensor:
        return torch.exp(output.float().clamp(max=30.0))


def build_loss(cfg: TrainConfig) -> nn.Module:
    if cfg.loss == "mse":
        return MSELoss()
    return TweedieLoss(cfg.tweedie_power)


def tweedie_deviance(y: np.ndarray, mu: np.ndarray, power: float) -> float:
    """Mean Tweedie deviance on the original scale (0 = perfect). Useful as an extra
    validation metric when comparing different values of ``power``."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-12, None)
    p = power
    if math.isclose(p, 0.0):
        dev = (y - mu) ** 2
    elif math.isclose(p, 1.0):
        dev = 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - y + mu)
    elif math.isclose(p, 2.0):
        dev = 2 * (np.log(mu / y) + y / mu - 1)
    else:
        dev = 2 * (np.power(np.clip(y, 0, None), 2 - p) / ((1 - p) * (2 - p))
                   - y * np.power(mu, 1 - p) / (1 - p) + np.power(mu, 2 - p) / (2 - p))
    return float(np.mean(dev))

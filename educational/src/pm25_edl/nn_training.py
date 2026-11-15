"""Train the six PM2.5 models through one batch interface.

Training labels are pm25_filled; validation targets are observed pm25. Keep
unlabeled dates as model inputs (especially for causal convolutions); apply
the label mask *before* computing any loss. Test data never enters fit_model.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def seed_everything(seed=88):
    """Call before constructing a model to reproduce its initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class FitHistory:
    train_loss: list = field(default_factory=list)
    val_nll: list = field(default_factory=list)
    best_epoch: int = 0              # 1-indexed; also usable for a fixed-epoch fit
    best_val_nll: float | None = None


def nig_nll(y, gamma, nu, alpha, beta):
    """Per-label Student-t predictive NLL; all inputs use the same target units.

    df=2*alpha; scale²=beta*(1+1/nu)/alpha. The output head guarantees
    nu,beta>0 and alpha>1. No likelihood is assigned to an individual source.
    """
    log_scale = 0.5 * (beta.log() - alpha.log() + torch.log1p(1 / nu))
    z = (y - gamma) * torch.exp(-log_scale)
    return (torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
            + 0.5 * (math.log(2 * math.pi) + alpha.log()) + log_scale
            + (alpha + 0.5) * torch.log1p(z.square() / (2 * alpha)))


def cosine_diversity_loss(deltas):
    """Reference pairwise absolute cosine penalty across source residuals."""
    columns = F.normalize(deltas.flatten(end_dim=-2), dim=0)
    i, j = torch.triu_indices(columns.shape[1], columns.shape[1], offset=1,
                              device=columns.device)
    return (columns.T @ columns)[i, j].abs().mean()


def evidential_loss(output, y, *, evidence_weight=0.01, lambda_sim=0.05,
                    use_cosine_similarity=True):
    """Mean Amini et al. evidential loss on finite normalized labels.

    Adds |y-gamma|*(2*nu+alpha) as an evidence penalty; the original cosine
    penalty is retained only for two-stage source residuals, and can be disabled
    with use_cosine_similarity=False. Single-stage outputs never use this term.
    Neither penalty is included in the reported predictive NLL.
    """
    if output.gamma.shape != y.shape:
        raise ValueError(f"Output shape {output.gamma.shape} does not match labels {y.shape}.")
    mask = torch.isfinite(y)
    if not mask.any():
        raise ValueError("No finite targets in this batch.")
    g, v, a, b = (x[mask] for x in (output.gamma, output.nu, output.alpha, output.beta))
    target = y[mask]
    loss = nig_nll(target, g, v, a, b).mean()
    if evidence_weight:
        loss = loss + evidence_weight * ((target - g).abs() * (2 * v + a)).mean()
    if use_cosine_similarity and lambda_sim and output.s_delta is not None:
        loss = loss + lambda_sim * cosine_diversity_loss(output.s_delta[mask])
    return loss


def _forward(model, batch, device):
    return model(*(x.to(device) for x in batch.inputs))


@torch.inference_mode()
def _validation_nll(model, split, batch_size, device):
    """Weighted over finite observed rows; includes the log-scale Jacobian."""
    model.eval()
    total, count = 0., 0
    for batch in split.batches(batch_size=batch_size):
        y = batch.y.to(device)
        mask = torch.isfinite(y)
        if not mask.any():
            continue
        out = _forward(model, batch, device)
        if out.gamma.shape != y.shape:
            raise ValueError("Model location and label shapes disagree.")
        nll = nig_nll(y[mask], out.gamma[mask], out.nu[mask], out.alpha[mask], out.beta[mask])
        total += nll.double().sum().item()
        count += nll.numel()
    if not count:
        raise ValueError("Validation has no finite observed pm25 labels.")
    value = total / count + math.log(model.target_scale.item())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite validation NLL; inspect the inputs and optimizer.")
    return value


def fit_model(model, train, validation=None, *, epochs, lr=1e-3, patience=10,
              batch_size=64, weight_decay=0., evidence_weight=0.01, lambda_sim=0.05,
              seed=88, device='cpu', optimizer_name='Adam', use_cosine_similarity=True):
    """Train on the provided training split; restore the lowest validation NLL.

    If validation=None, fit for exactly epochs and retain the final state.
    For a controlled Optuna comparison also call seed_everything(seed) before
    constructing each model. No optimizer state or weights are shared by trials.
    use_cosine_similarity=False disables the two-stage source penalty; it is
    always absent for single-stage models.
    """
    if train.role != 'train' or train.target_col != 'pm25_filled':
        raise ValueError("Training requires the train split with pm25_filled labels.")
    if validation is not None and (validation.role != 'val' or validation.target_col != 'pm25'):
        raise ValueError("Early stopping requires the val split with observed pm25 labels.")
    if epochs < 1 or batch_size < 1 or (validation is not None and (patience is None or patience < 1)):
        raise ValueError("epochs, batch_size and early-stopping patience must be positive.")
    if not (lr > 0 and weight_decay >= 0 and evidence_weight >= 0 and lambda_sim >= 0):
        raise ValueError("lr must be positive and loss/decay weights nonnegative.")
    optimizer_type = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}.get(optimizer_name)
    if optimizer_type is None:
        raise ValueError("optimizer_name must be Adam or AdamW.")
    seed_everything(seed)
    model.to(device)
    optimizer = optimizer_type(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator().manual_seed(seed)
    history, best_weights = FitHistory(), None
    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss, count = 0., 0
        for batch in train.batches(batch_size=batch_size, shuffle=True, generator=rng):
            y = batch.y.to(device)
            n = torch.isfinite(y).sum().item()
            if not n:
                continue
            optimizer.zero_grad(set_to_none=True)
            out = _forward(model, batch, device)
            loss = evidential_loss(out, y, evidence_weight=evidence_weight, lambda_sim=lambda_sim,
                                   use_cosine_similarity=use_cosine_similarity)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss; inspect the inputs and optimizer.")
            loss.backward()
            optimizer.step()
            weighted_loss += n * loss.item()
            count += n
        if not count:
            raise ValueError("Training has no finite pm25_filled labels.")
        history.train_loss.append(weighted_loss / count)
        if validation is None:
            history.best_epoch = epoch
            continue
        val = _validation_nll(model, validation, batch_size, device)
        history.val_nll.append(val)
        if history.best_val_nll is None or val < history.best_val_nll:
            history.best_val_nll, history.best_epoch = val, epoch
            best_weights = deepcopy(model.state_dict())
        elif epoch - history.best_epoch >= patience:
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()
    return history


@torch.inference_mode()
def predict_frame(model, split, *, batch_size=64, device='cpu'):
    """Predict every row, including missing labels; preserve input row order.

    Export NIG parameters in physical units: beta_raw=beta_scaled*target_scale².
    Corrected source columns are physical point estimates, not separate NIGs.
    """
    if split.role in ('val', 'test') and split.target_col != 'pm25':
        raise ValueError("Validation/test predictions must use observed pm25.")
    model.to(device).eval()
    n = len(split.frame)
    names = ['gamma', 'nu', 'alpha', 'beta']
    values = {name: np.empty(n, dtype=np.float64) for name in names}
    seen = np.zeros(n, dtype=bool)
    sources = None
    std = model.target_scale.item()
    for batch in split.batches(batch_size=batch_size):
        out = _forward(model, batch, device)
        ids = batch.row_ids.reshape(-1).cpu().numpy()
        if not (ids.size == out.gamma.numel() and (ids >= 0).all() and (ids < n).all()
                and len(np.unique(ids)) == ids.size and not seen[ids].any()):
            raise ValueError("Duplicate, invalid or misaligned row IDs in prediction batch.")
        seen[ids] = True
        for key, tensor in zip(names, (out.pred_physical, out.nu, out.alpha, out.beta)):
            values[key][ids] = tensor.reshape(-1).cpu().numpy()
        if out.s_corrected is not None:
            if sources is None:
                sources = np.empty((n, 3), dtype=np.float64)
            sources[ids] = out.s_corrected.reshape(-1, 3).cpu().numpy()
    if not n or not seen.all():
        raise ValueError("No data or omitted rows during prediction.")
    values['beta'] *= std**2
    if not all(np.isfinite(v).all() for v in values.values()):
        raise FloatingPointError("Non-finite NIG output.")
    if not ((values['alpha'] > 1).all() and (values['nu'] > 0).all() and (values['beta'] > 0).all()):
        raise FloatingPointError("Invalid NIG parameter domain.")
    if sources is not None:
        # The forward sum uses model precision. Re-sum the *exported* source
        # values in float64 so their exported total matches in that precision.
        total = sources.sum(axis=1)
        np.testing.assert_allclose(values['gamma'], total, rtol=2e-6, atol=1e-6)
        if not np.isfinite(total).all():
            raise FloatingPointError("Non-finite corrected source estimates.")
        values['gamma'] = total
    values['pred'] = values['gamma']
    aleatoric = values['beta'] / (values['alpha'] - 1)
    epistemic = aleatoric / values['nu']
    for name, variance in [('aleatoric', aleatoric), ('epistemic', epistemic),
                           ('total', aleatoric + epistemic)]:
        values[name + '_std'] = np.sqrt(variance)
    if sources is not None:
        for i, source in enumerate(('cmaq_pm25_bb', 'cmaq_pm25_anthro', 'cmaq_pm25_othr')):
            values['s_corrected_' + source] = sources[:, i]
    metadata = ['date', 'stcode', 'region5']
    if split.target_col is not None:
        metadata.append(split.target_col)
    raw = split.frame[metadata].reset_index(drop=True)
    return pd.concat([raw, pd.DataFrame(values)], axis=1)


def _self_test():
    from types import SimpleNamespace
    a = torch.full((2, 3), 2., dtype=torch.float64)
    g = torch.tensor([[1., -1., 2.], [34., 0., 4.]], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1.4, float('nan'), 1.1], [float('nan'), 0., 0.3]], dtype=g.dtype)
    ones = torch.ones_like(g)
    deltas = torch.randn(2, 3, 3, dtype=g.dtype)
    out = SimpleNamespace(gamma=g, nu=ones, alpha=a, beta=ones, s_delta=deltas)
    valid = torch.isfinite(y)
    ref = -torch.distributions.StudentT(4., g[valid], 1.).log_prob(y[valid]).mean()
    torch.testing.assert_close(evidential_loss(out, y, evidence_weight=0., lambda_sim=0.), ref)
    loss = evidential_loss(out, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(g.grad).all() and (g.grad[~valid] == 0).all()
    torch.testing.assert_close(cosine_diversity_loss(torch.eye(3)), torch.tensor(0.))
    torch.testing.assert_close(cosine_diversity_loss(torch.ones(4, 3)), torch.tensor(1.))
    torch.testing.assert_close(evidential_loss(out, y, use_cosine_similarity=False),
                               evidential_loss(out, y, lambda_sim=0.))
    single = SimpleNamespace(**{**vars(out), 's_delta': None})
    torch.testing.assert_close(evidential_loss(single, y, use_cosine_similarity=False),
                               evidential_loss(single, y, use_cosine_similarity=True))


if __name__ == '__main__':
    _self_test()
    print("Masked evidential loss, gradient, and source regularizer checks passed.")

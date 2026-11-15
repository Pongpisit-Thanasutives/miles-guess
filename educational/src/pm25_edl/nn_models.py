"""Six evidential PM2.5 regressors; tensors keep time before features.

The deterministic reference backbones are unchanged in design. A normal–inverse-
gamma (NIG) output replaces each point head. Two-stage models use the sum of
their corrected source concentrations for the NIG location; only the three
evidence parameters are learned by a separate head. The distribution describes
the total PM2.5 concentration, not three independently observed source targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


@dataclass
class EDLOutput:
    """One value per row or time step ([B] or [B,T]); last source axis is BB/anthro/othr."""

    gamma: Tensor                   # Predictive mean in standardized target space.
    nu: Tensor                      # Positive evidence for the latent mean.
    alpha: Tensor                   # Inverse-gamma shape > 1 (finite variance).
    beta: Tensor                    # Inverse-gamma scale, target-standardized units squared.
    pred_physical: Tensor           # Predicted mean in original concentration units.
    s_corrected: Tensor | None = None  # Nonnegative source concentrations, original units.
    s_delta: Tensor | None = None      # Unclamped residuals in source-standardized units.


def suggest_chain(trial, names, low=5, high=8):
    """Sample non-increasing powers of two; store actual widths with each trial."""
    widths = []
    for name in names:
        high = trial.suggest_int(f"{name}_exp", low, high)
        widths.append(2 ** high)
        trial.set_user_attr(name, widths[-1])
    return widths


class SharedMLP(nn.Module):
    """Three Linear → ReLU → Dropout blocks as in the reference MLP."""
    def __init__(self, n_features, h1, h2, hout, drop_rate):
        super().__init__()
        blocks = []
        for n_in, n_out in zip((n_features, h1, h2), (h1, h2, hout)):
            blocks.extend((nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(drop_rate)))
        self.layers = nn.Sequential(*blocks)

    def forward(self, x):
        return self.layers(x)


class SharedCNN1D(nn.Module):
    """Three causal temporal convolutions; 7, 5 and 3 taps cover 13 days."""
    def __init__(self, n_features, h1, h2, hout, drop_rate):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(n_features, h1, 7), nn.Conv1d(h1, h2, 5), nn.Conv1d(h2, hout, 3)
        ])
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B,T,F] -> [B,F,T] for Conv1d
        for conv in self.convs:
            x = F.pad(x, (conv.kernel_size[0] - 1, 0))  # past only; keep T
            x = self.dropout(F.relu(conv(x)))
        return x.transpose(1, 2)


class SharedDB(nn.Module):
    """Parallel causal CNN (dynamic features) and two-layer MLP (static features)."""
    def __init__(self, n_features, n_static, hts1, hts2, hts3, hstt1, hstt2,
                 drop_rate_ts, drop_rate_stt, fusion_width=None):
        super().__init__()
        if n_static < 1:
            raise ValueError("DB models require a separate static feature vector.")
        self.temporal = SharedCNN1D(n_features, hts1, hts2, hts3, drop_rate_ts)
        self.static = nn.Sequential(nn.Linear(n_static, hstt1), nn.ReLU(), nn.Dropout(drop_rate_stt),
                                    nn.Linear(hstt1, hstt2), nn.ReLU(), nn.Dropout(drop_rate_stt))
        # Linear on the last feature axis == 1×1 convolution at each date.
        self.fusion = (nn.Identity() if fusion_width is None
                       else nn.Linear(hts3 + hstt2, fusion_width))

    def forward(self, x, x_static):
        h_time = self.temporal(x)
        h_static = self.static(x_static)[:, None, :].expand(-1, x.shape[1], -1)
        return self.fusion(torch.cat((h_time, h_static), dim=-1))


class CorrectionHead(nn.Module):
    """Embed source k, join its embedding to the shared features, predict its residual."""
    def __init__(self, hout, hh1, hh2, hdrop_rate, hsk):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(1, hsk), nn.ReLU(), nn.Linear(hsk, hsk))
        self.correct = nn.Sequential(
            nn.Linear(hout + hsk, hh1), nn.ReLU(), nn.Dropout(hdrop_rate),
            nn.Linear(hh1, hh2), nn.ReLU(), nn.Dropout(hdrop_rate), nn.Linear(hh2, 1),
        )

    def forward(self, h, source):
        return self.correct(torch.cat((h, self.embed(source.unsqueeze(-1))), dim=-1)).squeeze(-1)


def _constant(value, name, size, positive=False):
    """Validate train-scaler statistics before turning them into checkpoint buffers."""
    t = torch.as_tensor(value, dtype=torch.float32).detach().reshape(-1)
    if t.numel() != size or not torch.isfinite(t).all() or (positive and not (t > 0).all()):
        raise ValueError(f"Invalid {name}: expected {size} finite" + (" positive" if positive else "") + " value(s)")
    return t.reshape(()) if size == 1 else t


class _EvidentialRegressor(nn.Module):
    """Apply the common output contract to a tabular, causal or dual backbone."""
    seq = False
    dual = False
    two_stage = False

    def __init__(self, backbone, width, target_mean, target_scale,
                 source_mean=None, source_scale=None, threshold=0.0, head_kwargs=None):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("target_mean", _constant(target_mean, "target_mean", 1))
        self.register_buffer("target_scale", _constant(target_scale, "target_scale", 1, positive=True))
        # MILES-GUESS LinearNormalGamma uses a spectrally normalized projection.
        self.nig_head = spectral_norm(nn.Linear(width, 3 if self.two_stage else 4))
        if self.two_stage:
            if source_mean is None or source_scale is None:
                raise ValueError("Two-stage source heads require the three TRAINING source means/scales.")
            self.register_buffer("source_mean", _constant(source_mean, "source_mean", 3))
            self.register_buffer("source_scale", _constant(source_scale, "source_scale", 3, positive=True))
            self.register_buffer("threshold", _constant(threshold, "threshold", 1))
            if self.threshold < 0:
                raise ValueError("threshold is the physical lower bound and must be nonnegative.")
            self.heads = nn.ModuleList(CorrectionHead(width, **head_kwargs) for _ in range(3))

    def forward(self, x: Tensor, x_static: Tensor | None = None) -> EDLOutput:
        if x.ndim != (3 if self.seq else 2):
            raise ValueError(f"{type(self).__name__} expects " + ("[B,T,F]." if self.seq else "[B,F]."))
        if self.dual:
            if x_static is None or x_static.ndim != 2 or x_static.shape[0] != x.shape[0]:
                raise ValueError("DB models require [B,S] static covariates alongside [B,T,F].")
            h = self.backbone(x, x_static)
        else:
            if x_static is not None:
                raise ValueError("For MLP/CNN1D, include static features in x.")
            h = self.backbone(x)
        raw = self.nig_head(h)
        corrected = delta = None
        if self.two_stage:
            # Preprocessing explicitly orders the first three standardized columns.
            # Each source has its own mean/scale. Its standardized values cannot
            # be summed directly, nor is z=0 a physical lower bound of zero.
            cmaq = x[..., :3]
            delta = torch.stack([head(h, cmaq[..., k]) for k, head in enumerate(self.heads)], dim=-1)
            corrected = ((cmaq + delta) * self.source_scale + self.source_mean).clamp_min(self.threshold)
            pred = corrected.sum(dim=-1)  # physical totals and exports share this exact sum
            gamma = (pred - self.target_mean) / self.target_scale
        else:
            gamma, raw = raw[..., 0], raw[..., 1:]
            pred = gamma * self.target_scale + self.target_mean

        # A margin prevents softplus underflow or rounding alpha to exactly one.
        nu, a, beta = (F.softplus(raw) + torch.finfo(raw.dtype).eps).unbind(dim=-1)
        return EDLOutput(gamma, nu, 1.0 + a, beta, pred, corrected, delta)


class MLP(_EvidentialRegressor):
    """Tabular EDL model. x: [B, all features]."""
    def __init__(self, n_features, h1=64, h2=32, hout=32, drop_rate=0.1,
                 target_mean=0., target_scale=1.):
        super().__init__(SharedMLP(n_features, h1, h2, hout, drop_rate), hout, target_mean, target_scale)


class CNN1D(_EvidentialRegressor):
    """Causal EDL model. x: [B,T, all features], including repeated static features."""
    seq = True

    def __init__(self, n_features, h1=64, h2=32, hout=32, drop_rate=0.1,
                 target_mean=0., target_scale=1.):
        super().__init__(SharedCNN1D(n_features, h1, h2, hout, drop_rate), hout, target_mean, target_scale)


class DB(_EvidentialRegressor):
    """Dual-branch EDL model. x: [B,T, dynamic features]; x_static: [B,7]."""
    seq = dual = True

    def __init__(self, n_features, n_static, hts1=64, hts2=32, hts3=32, hstt1=32, hstt2=16,
                 drop_rate_ts=0.1, drop_rate_stt=0.1, target_mean=0., target_scale=1.):
        backbone = SharedDB(n_features, n_static, hts1, hts2, hts3, hstt1, hstt2,
                            drop_rate_ts, drop_rate_stt)
        super().__init__(backbone, hts3 + hstt2, target_mean, target_scale)


class TwoStageMLP(_EvidentialRegressor):
    """Tabular backbone and three CMAQ residual heads."""
    two_stage = True

    def __init__(self, n_features, source_mean, source_scale, h1=64, h2=32, hout=32,
                 drop_rate=0.1, hh1=32, hh2=16, hsk=16, hdrop_rate=0.1,
                 target_mean=0., target_scale=1., threshold=0.):
        super().__init__(SharedMLP(n_features, h1, h2, hout, drop_rate), hout, target_mean,
                         target_scale, source_mean, source_scale, threshold,
                         dict(hh1=hh1, hh2=hh2, hsk=hsk, hdrop_rate=hdrop_rate))


class TwoStageCNN1D(_EvidentialRegressor):
    """Causal CNN backbone and three CMAQ residual heads per date."""
    seq = two_stage = True

    def __init__(self, n_features, source_mean, source_scale, h1=64, h2=32, hout=32,
                 drop_rate=0.1, hh1=32, hh2=16, hsk=16, hdrop_rate=0.1,
                 target_mean=0., target_scale=1., threshold=0.):
        super().__init__(SharedCNN1D(n_features, h1, h2, hout, drop_rate), hout, target_mean,
                         target_scale, source_mean, source_scale, threshold,
                         dict(hh1=hh1, hh2=hh2, hsk=hsk, hdrop_rate=hdrop_rate))


class TwoStageDB(_EvidentialRegressor):
    """Dual backbone with the reference 1×1 fusion before its three correction heads."""
    seq = dual = two_stage = True

    def __init__(self, n_features, n_static, source_mean, source_scale, hts1=64, hts2=32, hts3=32,
                 hstt1=32, hstt2=16, hout=32, drop_rate_ts=0.1, drop_rate_stt=0.1, hh1=32,
                 hh2=16, hsk=16, hdrop_rate=0.1, target_mean=0., target_scale=1., threshold=0.):
        backbone = SharedDB(n_features, n_static, hts1, hts2, hts3, hstt1, hstt2,
                            drop_rate_ts, drop_rate_stt, fusion_width=hout)
        super().__init__(backbone, hout, target_mean, target_scale, source_mean, source_scale, threshold,
                         dict(hh1=hh1, hh2=hh2, hsk=hsk, hdrop_rate=hdrop_rate))


MODELS = {c.__name__: c for c in (MLP, CNN1D, DB, TwoStageMLP, TwoStageCNN1D, TwoStageDB)}


def build_model(name, n_features, n_static=0, *, feature_option=None, source_mean=None,
                source_scale=None, target_mean=0., target_scale=1., **architecture):
    """Construct a named model; pass the dimensions/statistics from the fitted preprocessor."""
    key = str(name).lower().replace('_', '').replace('-', '')
    try:
        cls = {n.lower(): c for n, c in MODELS.items()}[key]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose one of {list(MODELS)}.") from None
    if feature_option is not None and feature_option not in ('baseline', 'at-bb', 'at-bb-aod'):
        raise ValueError(f"Unknown CMAQ feature option: {feature_option!r}.")
    if cls.two_stage:
        if feature_option == 'baseline' or n_features < 3:
            raise ValueError("A two-stage model needs the three distinct CMAQ sources; use at-bb or at-bb-aod.")
        architecture.update(source_mean=source_mean, source_scale=source_scale)
    if cls.dual:
        architecture['n_static'] = n_static
    elif n_static:
        raise ValueError("n_static is separate only for DB. MLP/CNN1D include static columns in x.")
    model = cls(n_features, target_mean=target_mean, target_scale=target_scale, **architecture)
    return model


def self_test():
    """All six shapes, valid Student-t likelihood, missing labels, source conservation, causality."""
    torch.manual_seed(88)
    for name, cls in MODELS.items():
        dim = 8 if cls.dual else 15  # 3 sources + AOD + 4 WRF, and 7 static if not separate
        model = build_model(name, dim, n_static=7 if cls.dual else 0, feature_option='at-bb-aod',
                            source_mean=[2, 5, 7], source_scale=[0.5, 1, 3], target_mean=13, target_scale=4)
        model.eval()
        x = torch.randn((2, 6, dim) if cls.seq else (2, dim))
        static = torch.randn(2, 7) if cls.dual else None
        o = model(x, static)
        assert o.gamma.shape == x.shape[:-1] and torch.all(o.alpha > 1)
        assert torch.all(o.nu > 0) and torch.all(o.beta > 0)
        if cls.two_stage:
            assert o.s_corrected.shape == (*x.shape[:-1], 3)
            assert torch.all(o.s_corrected >= 0) and torch.equal(o.s_corrected.sum(-1), o.pred_physical)
        torch.testing.assert_close(o.gamma * 4 + 13, o.pred_physical)
        y = torch.randn_like(o.gamma)
        y.reshape(-1)[0] = torch.nan
        observed = y.isfinite()
        scale = (o.beta * (1 + o.nu) / (o.nu * o.alpha)).sqrt()
        distribution = torch.distributions.StudentT(2 * o.alpha[observed], o.gamma[observed], scale[observed])
        loss = -distribution.log_prob(y[observed]).mean()
        assert observed.sum() == y.numel() - 1 and torch.isfinite(loss)
        loss.backward()
        if cls.seq:
            changed = x.clone()
            changed[:, 3:, :] += 10
            later = model(changed, static)
            for field in ("gamma", "nu", "alpha", "beta", "pred_physical"):
                torch.testing.assert_close(getattr(o, field)[:, :3], getattr(later, field)[:, :3])


if __name__ == '__main__':
    self_test()
    print("Six EDL forward passes and model checks passed.")

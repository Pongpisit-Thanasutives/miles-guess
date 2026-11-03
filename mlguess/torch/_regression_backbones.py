"""Backbones and source accounting shared by the scalar Torch regression models."""
from operator import index

import torch
from torch import nn
from torch.nn import functional as F

from .checkpoint import load_state_dict
from .layers import LinearNormalGamma
from .mc_dropout import as_model_input, inference_context, _output_moments, _scale_moments


def _width(value):
    if isinstance(value, bool) or index(value) < 1:
        raise ValueError("Feature counts, kernel sizes and hidden widths must be positive.")
    return index(value)


def _mlp(in_features, widths, dropout):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(_width(in_features), _width(width)), nn.ReLU(), nn.Dropout(dropout)))
        in_features = width
    return nn.Sequential(*layers)


class CausalEncoder(nn.Module):
    """Three left-padded convolutions preserve sequence length and temporal causality."""

    def __init__(self, in_features, widths, dropout, kernel_sizes):
        super().__init__()
        self.in_features = _width(in_features)
        self.kernel_sizes = tuple(_width(k) for k in kernel_sizes)
        if len(self.kernel_sizes) != 3:
            raise ValueError("kernel_sizes must contain exactly three positive integers.")
        self.convs = nn.ModuleList()
        for width, kernel in zip(widths, self.kernel_sizes):
            self.convs.append(nn.Conv1d(_width(in_features), _width(width), kernel))
            in_features = width
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != self.in_features or x.shape[1] == 0:
            raise ValueError(f"Expected [batch, positive time, {self.in_features}] input.")
        x = x.transpose(1, 2)
        for conv, k in zip(self.convs, self.kernel_sizes):
            x = self.dropout(F.relu(conv(F.pad(x, (k - 1, 0)))))
        return x.transpose(1, 2)


class DualEncoder(nn.Module):
    """Concatenate causal temporal features and time-broadcast static features."""

    def __init__(self, n_fts, n_fstatic, temporal_widths, static_widths,
                 drop_rate_ts, drop_rate_stt, kernel_sizes, fusion_width=None):
        super().__init__()
        self.temporal = CausalEncoder(n_fts, temporal_widths, drop_rate_ts, kernel_sizes)
        self.static = _mlp(n_fstatic, static_widths, drop_rate_stt)
        self.n_fstatic = _width(n_fstatic)
        joint_width = temporal_widths[-1] + static_widths[-1]
        # A Linear over the last axis is a 1x1 temporal convolution with the
        # same bias. The two-stage DB uses a fusion layer before its source heads.
        self.fusion = (nn.Identity() if fusion_width is None
                       else nn.Linear(joint_width, _width(fusion_width)))

    def forward(self, x_ts, x_static):
        h = self.temporal(x_ts)
        if x_static.ndim != 2 or x_static.shape != (h.shape[0], self.n_fstatic):
            raise ValueError(f"Expected static input of shape [batch, {self.n_fstatic}].")
        static = self.static(x_static).unsqueeze(1).expand(-1, h.shape[1], -1)
        return self.fusion(torch.cat((h, static), dim=-1))


class CorrectionHead(nn.Module):
    """Independent residual head consuming shared features and one embedded source."""

    def __init__(self, hout, hh1, hh2, dropout, hsk):
        super().__init__()
        self.source_embed = nn.Sequential(nn.Linear(1, _width(hsk)), nn.ReLU(), nn.Linear(hsk, hsk))
        self.hidden = _mlp(hout + hsk, (hh1, hh2), dropout)
        self.out = nn.Linear(hh2, 1)

    def forward(self, shared, source):
        embedded = self.source_embed(source.unsqueeze(-1))
        return self.out(self.hidden(torch.cat((shared, embedded), dim=-1))).squeeze(-1)


def _buffer_values(value, size, name):
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().reshape(-1).clone()
    if tensor.numel() != size or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain {size} finite value(s).")
    return tensor


class ScalarHead(nn.Module):
    """Predict scalar NIG parameters, optionally taking the location from a source sum."""

    def __init__(self, width, evidential, source_location=False):
        super().__init__()
        self.evidential = evidential
        self.source_location = source_location
        if source_location:
            # No independent location parameter: gamma must equal the source sum.
            self.projection = nn.Linear(width, 3) if evidential else None
        elif evidential:
            self.projection = LinearNormalGamma(width, 1, spectral_norm=False)
        else:
            self.projection = nn.Linear(width, 1)

    def forward(self, hidden, location=None):
        if not self.evidential:
            return location if self.source_location else self.projection(hidden).squeeze(-1)
        if self.source_location:
            logits = self.projection(hidden).reshape(*location.shape, 3)
            nu, alpha_m1, beta = F.softplus(logits).unbind(-1)
            alpha = 1 + alpha_m1
        else:
            # LinearNormalGamma expects examples along its first dimension.
            params = self.projection(hidden.reshape(-1, hidden.shape[-1]))
            location, nu, alpha, beta = (p.reshape(hidden.shape[:-1]) for p in params)
        # Softplus is positive in exact arithmetic. Guard underflow and the
        # rounding of 1 + tiny evidence to exactly 1 (undefined NIG variance).
        eps = torch.finfo(nu.dtype).eps
        return location, nu.clamp_min(eps), alpha.clamp_min(1 + eps), beta.clamp_min(eps)


class ScalarRegressionModel(nn.Module):
    """Scalar predictive moments with the same affine variance convention as DNN."""
    two_stage = False
    sequential = True

    def predict(self, *inputs, y_scaler=None, return_uncertainties=True):
        """Run in eval mode and restore the original module states afterwards.

        With uncertainties enabled the prediction is (mean, aleatoric variance,
        epistemic variance, total variance), optionally in target physical units.
        Otherwise it is the raw NIG tuple or point tensor in target coordinates.
        Two-stage models always retain (prediction, corrected, delta, hidden);
        the latter three tensors retain their forward-pass coordinates.
        """
        if return_uncertainties and not self.evidential:
            raise ValueError("Use return_uncertainties=False for a point model.")
        with inference_context(self):
            output = self(*(as_model_input(x, self) for x in inputs))
            if not return_uncertainties:
                return output
            params = output[0] if self.two_stage else output
            moments = self.predict_uncertainty(params, y_scaler=y_scaler)
            return (moments, *output[1:]) if self.two_stage else moments

    def predict_uncertainty(self, params, y_scaler=None):
        """Return DNN-style moments for per-row or per-timestep NIG parameters."""
        if not self.evidential:
            raise ValueError("Evidential parameters are required to compute uncertainty.")
        parts = tuple(params)
        if len(parts) != 4 or not all(isinstance(p, torch.Tensor) for p in parts):
            raise ValueError("Expected a tuple of four scalar NIG parameter tensors.")
        shape = parts[0].shape
        if any(p.shape != shape for p in parts):
            raise ValueError("All NIG parameter tensors must have the same shape.")
        flat = tuple(p.reshape(-1, 1) for p in parts)
        mean, ale, epi, _ = _output_moments(flat, "nig")
        mean, ale, epi = _scale_moments(mean, ale, epi, y_scaler)
        moments = mean, ale, epi, ale + epi
        if not all(torch.isfinite(p).all() for p in moments):
            raise ValueError("Predictive uncertainty moments must be finite.")
        return tuple(p.reshape(shape) for p in moments)

    def load_weights(self, weights_path, *, weights_only=True):
        """Restore a state dict or trainer checkpoint, including source scaler buffers."""
        checkpoint = load_state_dict(weights_path, weights_only=weights_only)
        self.load_state_dict(checkpoint.get("model_state_dict", checkpoint))


class SourceModel(ScalarRegressionModel):
    """Correct source inputs; sum in physical units, then transform to target scale.

    The model assumes sources occupy the first n_sources input channels. Each
    correction head produces a residual in that source's input coordinates.
    Standardized inputs require all four affine scaling statistics from training.
    Without statistics, sources and target are assumed to use the same raw units.
    A supplied threshold is a lower bound in *source input coordinates*. With
    threshold=None, the lower bound is zero in raw units for each source.

    Forward returns (prediction, s_corrected, s_delta, hidden). The returned
    s_delta is the head's proposal before clamping, matching the reference
    diversity loss. Effective physical corrections are corrected minus original.
    """
    two_stage = True

    def __init__(self, n_sources, n_features, hout, hh1, hh2, hdrop_rate, hsk,
                 threshold, source_mean, source_scale, target_mean, target_scale, evidential):
        super().__init__()
        self.n_sources = _width(n_sources)
        if self.n_sources > n_features:
            raise ValueError("n_sources must not exceed the number of source-bearing input features.")
        provided = tuple(v is not None for v in (source_mean, source_scale, target_mean, target_scale))
        if any(provided) and not all(provided):
            raise ValueError("Provide source_mean, source_scale, target_mean and target_scale together.")
        if not any(provided):
            source_mean, source_scale = torch.zeros(n_sources), torch.ones(n_sources)
            target_mean, target_scale = 0.0, 1.0
        for name, val, size in (("source_mean", source_mean, n_sources),
                                ("source_scale", source_scale, n_sources),
                                ("target_mean", target_mean, 1), ("target_scale", target_scale, 1)):
            data = _buffer_values(val, size, name)
            if name.endswith("_scale") and (data <= 0).any():
                raise ValueError(f"{name} must be positive.")
            self.register_buffer(name, data)
        if threshold is None:
            lower = -self.source_mean / self.source_scale
        else:
            lower = torch.as_tensor(threshold, dtype=torch.float32).reshape(-1)
            if lower.numel() == 1:
                lower = lower.expand(n_sources)
        self.register_buffer("threshold", _buffer_values(lower, n_sources, "threshold"))
        self.heads = nn.ModuleList(
            CorrectionHead(hout, hh1, hh2, hdrop_rate, hsk) for _ in range(n_sources)
        )
        self.evidential = evidential
        self.output_head = ScalarHead(hout, evidential, source_location=True)

    def _source_output(self, x, hidden):
        original = x[..., :self.n_sources]
        proposed = torch.stack([head(hidden, original[..., k])
                                for k, head in enumerate(self.heads)], dim=-1)
        corrected = torch.maximum(original + proposed, self.threshold)
        raw = corrected * self.source_scale + self.source_mean
        location = (raw.sum(dim=-1, keepdim=not self.sequential) - self.target_mean) / self.target_scale
        return self.output_head(hidden, location), corrected, proposed, hidden

"""Properties of the causal encoders, source conservation and scalar NIG heads."""
import pytest

torch = pytest.importorskip("torch")
from sklearn.preprocessing import StandardScaler
from mlguess.torch.models import CNN1D, DB, TwoStageMLP, TwoStageCNN1D, TwoStageDB


def make_model(cls, **kwargs):
    widths = dict(h1=6, h2=5, hout=4, drop_rate=0.1)
    dual = dict(n_fts=3, n_fstatic=2, hts1=6, hts2=5, hts3=4, hstt1=3, hstt2=2,
                drop_rate_ts=0.1, drop_rate_stt=0.2)
    heads = dict(n_sources=2, hh1=5, hh2=4, hdrop_rate=0.1, hsk=3)
    if cls is CNN1D:
        return cls(in_features=3, **widths, **kwargs)
    if cls is DB:
        return cls(**dual, **kwargs)
    if cls is TwoStageDB:
        return cls(hout=4, **dual, **heads, **kwargs)
    return cls(n_fkey=2, n_fcontrol=1, **widths, **heads, **kwargs)


SCALING = dict(source_mean=[1., 7.], source_scale=[3., 2.],
               target_mean=11., target_scale=4.)


@pytest.mark.parametrize("cls", [CNN1D, DB, TwoStageCNN1D, TwoStageDB])
def test_changes_to_future_observations_do_not_affect_past_predictions(cls):
    torch.manual_seed(17)
    model = make_model(cls).eval()
    x = torch.randn(2, 17, 3)
    changed = x.clone()
    changed[:, 9:] += 100 * torch.randn_like(changed[:, 9:])
    static = (torch.randn(2, 2),) if cls in (DB, TwoStageDB) else ()
    out1, out2 = model(x, *static), model(changed, *static)
    if model.two_stage:
        for first, second in zip(out1[1:], out2[1:]):
            torch.testing.assert_close(first[:, :9], second[:, :9])
        out1, out2 = out1[0], out2[0]
    for first, second in zip(out1, out2):
        assert first.shape == (2, 17)
        torch.testing.assert_close(first[:, :9], second[:, :9])


@pytest.mark.parametrize("cls", [TwoStageMLP, TwoStageCNN1D, TwoStageDB])
@pytest.mark.parametrize("evidential", [False, True])
def test_source_location_is_clamped_sum_in_physical_units(cls, evidential):
    model = make_model(cls, evidential=evidential, **SCALING).eval()
    # With zero residuals these channels imply raw contributions -5 and 9.
    # Clamping at raw zero gives predicted total 9; its target coordinate is -0.5.
    for head in model.heads:
        for p in head.parameters():
            torch.nn.init.zeros_(p)
    x = torch.tensor([[-2., 1., -1.], [-2., 1., 1.]])
    if model.sequential:
        x = x[:, None, :].repeat(1, 3, 1)
    static = (torch.zeros(2, 2),) if cls is TwoStageDB else ()
    prediction, corrected, proposed, hidden = model(x, *static)
    location = prediction[0] if evidential else prediction
    assert location.shape == ((2, 3) if model.sequential else (2, 1))
    torch.testing.assert_close(location, torch.full_like(location, -0.5))
    raw = corrected * model.source_scale + model.source_mean
    torch.testing.assert_close(raw[..., 0], torch.zeros_like(raw[..., 0]))
    torch.testing.assert_close(raw[..., 1], torch.full_like(raw[..., 1], 9.))
    torch.testing.assert_close(proposed, torch.zeros_like(proposed))
    assert hidden.shape[:-1] == raw.shape[:-1]


@pytest.mark.parametrize("cls", [CNN1D, DB, TwoStageMLP, TwoStageCNN1D, TwoStageDB])
def test_predict_reports_target_scaled_nig_moments(cls):
    model = make_model(cls)
    x = torch.ones(2, 4, 3) if model.sequential else torch.ones(2, 3)
    static = (torch.ones(2, 2),) if cls in (DB, TwoStageDB) else ()
    scaler = StandardScaler().fit([[3.], [7.]])
    out = model.predict(x, *static, return_uncertainties=False)
    params = out[0] if model.two_stage else out
    gamma, nu, alpha, beta = params
    out = model.predict(x, *static, y_scaler=scaler)
    mean, ale, epi, total = out[0] if model.two_stage else out
    assert model.training
    torch.testing.assert_close(mean, 5 + 2 * gamma)
    torch.testing.assert_close(ale, 4 * beta / (alpha - 1))
    torch.testing.assert_close(epi, ale / nu)
    torch.testing.assert_close(total, ale + epi)
    assert not mean.requires_grad


def test_partial_or_invalid_source_scale_rejected():
    with pytest.raises(ValueError, match="together"):
        make_model(TwoStageDB, source_mean=[1., 2.])
    with pytest.raises(ValueError, match="positive"):
        make_model(TwoStageCNN1D, **{**SCALING, "source_scale": [1., 0.]})
    with pytest.raises(ValueError, match="threshold"):
        make_model(TwoStageMLP, threshold=[0., 0., 0.])

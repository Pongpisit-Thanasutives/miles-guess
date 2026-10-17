"""Self-contained CPU example: trained NIG ensemble and MC-dropout moments.

Run: python examples/uncertainty_quickstart.py
This checks implementation identities on synthetic data, not calibration.
"""
import json
import platform

import numpy as np
import torch

from mlguess.torch.models import DNN, CategoricalDNN, seed_everything
from mlguess.torch.regression_losses import LipschitzMSELoss
from mlguess.uncertainty import ensemble_uncertainties


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(204)
    x = torch.tensor(rng.normal(size=(64, 2)), dtype=torch.float32)
    y = .5 * x[:, :1] - .2 * x[:, 1:] + torch.tensor(
        rng.normal(scale=.1, size=(64, 1)), dtype=torch.float32)
    members = []
    losses = []
    for seed in [17, 29]:
        seed_everything(seed)
        model = DNN(2, 1, layer_size=[16], dr=[.1], lng=True).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        criterion = LipschitzMSELoss(coef=.01)
        values = []
        for _ in range(60):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(*model(x), y)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            values.append(loss.item())
        assert np.mean(values[-5:]) < np.mean(values[:5])
        members.append(model)
        losses.append([values[0], values[-1]])
    moments = [member.predict(x) for member in members]
    means, aleatoric, epistemic, totals = [
        np.stack([values[k].cpu().numpy() for values in moments]) for k in range(4)
    ]
    mixture = ensemble_uncertainties(means, aleatoric, epistemic)
    # Independent second-moment identity for the actual predicted distributions.
    reference = (totals.astype(float) + means.astype(float)**2).mean(axis=0) - means.astype(float).mean(axis=0)**2
    np.testing.assert_allclose(mixture['total'], reference, rtol=2e-6)
    dropout = members[0].predict_dropout(x, mc_forward_passes=12, return_dict=True)
    np.testing.assert_allclose(dropout['epistemic'], dropout['within_epistemic'] + dropout['between_epistemic'])
    np.testing.assert_allclose(dropout['total'], dropout['aleatoric'] + dropout['epistemic'])
    classifier = CategoricalDNN(2, 3, layer_size=[8], dr=[.3]).cpu()
    categorical = classifier.predict_dropout(x, mc_forward_passes=12, return_dict=True)
    probability = categorical['mean']
    np.testing.assert_allclose(categorical['total'], probability*(1-probability), rtol=2e-6, atol=1e-7)
    point_model = DNN(2, 1, layer_size=[8], dr=[.3]).cpu()
    point = point_model.predict_dropout(x, mc_forward_passes=12, return_dict=True)
    assert point['aleatoric'] is None and point['total'] is None
    print(json.dumps(dict(python=platform.python_version(), torch=torch.__version__,
                         numpy=np.__version__, device='cpu', losses=losses,
                         ensemble_shape=list(mixture['total'].shape),
                         mean_aleatoric=float(mixture['aleatoric'].mean()),
                         mean_within_epistemic=float(mixture['within_epistemic'].mean()),
                         mean_between_epistemic=float(mixture['between_epistemic'].mean()),
                         checks='NIG mixture, evidential dropout, Dirichlet dropout, point-noise unknown: passed'), indent=2))


if __name__ == '__main__':
    main()

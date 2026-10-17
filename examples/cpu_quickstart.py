"""Train, evaluate, and reload a small evidential regressor on CPU.

Run from the unpacked project: python examples/cpu_quickstart.py
No external data, GPU, TensorFlow, or optional tuning packages are required.
"""
from pathlib import Path
import tempfile

import numpy as np
import torch

from mlguess.torch.models import DNN, seed_everything
from mlguess.torch.regression_losses import LipschitzMSELoss
from mlguess.torch.checkpoint import load_state_dict


def main():
    seed_everything(42)
    torch.set_num_threads(2)
    device = torch.device("cpu")
    rng = np.random.default_rng(42)
    x = torch.from_numpy(rng.normal(size=(128, 3)).astype(np.float32)).to(device)
    y = 0.7 * x[:, :1] - 0.2 * x[:, 1:2] + 0.05 * torch.randn(128, 1)
    kwargs = dict(input_size=3, output_size=1, layer_size=[16, 16], dr=[0.0, 0.0], lng=True)
    model = DNN(**kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = LipschitzMSELoss(coef=0.01)
    losses = []
    for _ in range(40):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(*model(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    mu, aleatoric, epistemic, total = model.predict(x, return_uncertainties=True)
    assert torch.isfinite(mu).all()
    np.testing.assert_allclose(total, aleatoric + epistemic)
    assert losses[-1] < losses[0]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = DNN(**kwargs).to(device)
        restored.load_state_dict(load_state_dict(path))
        restored.eval()
        np.testing.assert_allclose(restored.predict(x)[0], mu)
    print(f"Python CPU quickstart passed: torch={torch.__version__}, numpy={np.__version__}")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}; predictions: {mu.shape}")
    print("Aleatoric, epistemic, and total uncertainty are variances.")


if __name__ == "__main__":
    main()

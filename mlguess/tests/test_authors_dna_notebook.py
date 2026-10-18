"""Validate original DNA notebook helpers with mocked BERT interfaces.

These tests do not download/train BERT or claim biological benchmark results.
The small fixture rows exercise evaluation and plotting plumbing only.
"""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler

from mlguess.evaluation import evaluate_classifier, evaluate_nig
from mlguess.uncertainty import dirichlet_uncertainties, nig_uncertainties, inverse_scale_moments
from mlguess.torch.class_losses import relu_evidence, edl_digamma_loss, edl_log_loss, edl_mse_loss

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "notebooks/dna_example_torch.ipynb"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="Source notebook not installed")


def code_cells():
    notebook = json.loads(PATH.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture
def notebook_namespace(monkeypatch):
    calls = []

    class FakeBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=4, max_position_embeddings=16)
            self.embedding = nn.Embedding(16, 4)
            self.dropout = nn.Dropout(.8)
            self.last_token_type_ids = None

        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return cls()

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            self.last_token_type_ids = token_type_ids
            return SimpleNamespace(last_hidden_state=self.dropout(self.embedding(input_ids)))

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls.append((name, kwargs))
            return cls()

        def __call__(self, sequences, **kwargs):
            if isinstance(sequences, str):
                sequences = [sequences]
            tokens = torch.tensor([[ord(sequence[0]) % 16, len(sequence) % 16] for sequence in sequences])
            return dict(input_ids=tokens, attention_mask=torch.ones_like(tokens), token_type_ids=torch.zeros_like(tokens))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(BertModel=FakeBert, BertTokenizer=FakeTokenizer))
    namespace = dict(np=np, pd=pd, plt=plt, mdates=mdates, torch=torch, nn=nn, Path=Path,
                     contextmanager=contextmanager, PRETRAINED_NAME="bert-base-uncased",
                     LOCAL_FILES_ONLY=True, DEVICE=torch.device("cpu"), tol=1e-8,
                     relu_evidence=relu_evidence, edl_digamma_loss=edl_digamma_loss,
                     edl_log_loss=edl_log_loss, edl_mse_loss=edl_mse_loss,
                     dirichlet_uncertainties=dirichlet_uncertainties, nig_uncertainties=nig_uncertainties,
                     inverse_scale_moments=inverse_scale_moments, evaluate_classifier=evaluate_classifier,
                     evaluate_nig=evaluate_nig, display=lambda obj: None, _calls=calls)
    for cell in code_cells():
        module = ast.parse(cell)
        definitions = [node for node in module.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        if definitions:
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PATH), "exec"), namespace)
    return namespace


def test_original_configuration_and_loss_definitions_retained():
    notebook = json.loads(PATH.read_text())
    source = "\n".join(code_cells())
    assert 'PRETRAINED_NAME = "bert-base-uncased"' in source
    assert 'dna_sequence = "AGCTAGCTAGCT"' in source
    assert 'num_classes = 10' in source
    assert 'annealing_coefficient = 10.' in source
    assert 'n_tasks=1, training_var=[0.0033433]' in source
    assert '[[1.45456457]]' in source
    assert 'make_classification' not in source
    assert 'base_frequency' not in source
    original = json.loads((ROOT / "notebooks/legacy/dna_example_torch_original.ipynb").read_text())
    expected = "".join(original["cells"][28]["source"]).replace("pi = torch.tensor(np.pi)", "pi = nu.new_tensor(np.pi)").strip()
    assert expected in source
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_classifier_original_interface_and_deterministic_moments(notebook_namespace):
    ns = notebook_namespace
    model = ns["DNABert"](10).train()
    model.bert.dropout.eval()
    states = [module.training for module in model.modules()]
    inputs = ns["load_tokenizer"]()(["AGCTAGCTAGCT", "CG"])
    first = model.predict_uncertainty(**inputs)
    second = model.predict_uncertainty(**inputs)
    assert first[0].shape == (2, 10)
    for actual, repeated in zip(first, second):
        np.testing.assert_array_equal(actual, repeated)
    probability, dst, aleatoric, epistemic = first
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    np.testing.assert_allclose(aleatoric + epistemic, probability * (1 - probability))
    assert (dst > 0).all() and (dst <= 1).all()
    assert [module.training for module in model.modules()] == states
    assert model.bert.last_token_type_ids is inputs["token_type_ids"]
    assert all(name == "bert-base-uncased" and kwargs == {"local_files_only": True} for name, kwargs in ns["_calls"])


def test_regression_scaling_and_preserved_metadata(notebook_namespace):
    ns = notebook_namespace
    model = ns["DNABertRegressor"](1, [0.0033433]).train()
    inputs = ns["load_tokenizer"]()(["AGCTAGCTAGCT", "CG"])
    raw = model.predict_uncertainty(**inputs)
    scaler = StandardScaler().fit([[10.], [12.], [14.]])
    physical = model.predict_uncertainty(**inputs, y_scaler=scaler)
    assert model.training
    assert model.training_var == [0.0033433]
    np.testing.assert_allclose(physical[0], scaler.inverse_transform(raw[0]))
    np.testing.assert_allclose(physical[1], raw[1] * scaler.scale_**2)
    np.testing.assert_allclose(physical[2], raw[2] * scaler.scale_**2)


@pytest.mark.parametrize("value", [-10000., 10000.])
def test_nig_head_numerical_extremes(notebook_namespace, value):
    head = notebook_namespace["LinearNormalGamma"](4, 1)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias[:] = torch.tensor([0, value, value, value])
    mu, nu, alpha, beta = head(torch.zeros(2, 4))
    assert torch.isfinite(torch.stack([mu, nu, alpha, beta])).all()
    assert (nu > 0).all() and (alpha > 1).all() and (beta > 0).all()


def test_batch_predictions_order_shape_and_checkpoint_roundtrip(notebook_namespace, tmp_path):
    ns = notebook_namespace
    tokenizer = ns["load_tokenizer"]()
    sequences = ["AGCT", "CG", "GGG", "T"]
    model = ns["DNABertRegressor"](1, [0.0033433]).train()
    many = ns["predict_batches"](model, tokenizer, sequences, batch_size=2)
    single = ns["predict_batches"](model, tokenizer, sequences, batch_size=1)
    assert model.training
    for a, b in zip(many, single):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
        assert a.shape == (4, 1)
    path = tmp_path / "state.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    restored = ns["restore_trained_state"](ns["DNABertRegressor"](1, [0.0033433]), path)
    assert not restored.training
    again = ns["predict_batches"](restored, tokenizer, sequences)
    for a, b in zip(many, again):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("sequences,batch_size", [([], 1), ([""], 1), ([None], 1), (["AT"], 0), (["AT"], True)])
def test_invalid_prediction_inputs(notebook_namespace, sequences, batch_size):
    ns = notebook_namespace
    with pytest.raises(ValueError):
        ns["predict_batches"](ns["DNABert"](10), ns["load_tokenizer"](), sequences, batch_size)


def test_original_loss_combination_gradients(notebook_namespace):
    ns = notebook_namespace
    head = ns["LinearNormalGamma"](4, 1)
    gamma, nu, alpha, beta = head(torch.randn(2, 4))
    target = gamma.new_tensor([[1.45456457], [1.45456457]])
    loss = ns["EvidentialMarginalLikelihood"]()(gamma, nu, alpha, beta, target)
    loss += ns["EvidenceRegularizer"]()(gamma, nu, alpha, target)
    loss += ns["modified_mse"](gamma, nu, alpha, beta, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(head.linear.weight.grad).all()
    for name in ["digamma", "log", "mse"]:
        logits = torch.randn(2, 10, requires_grad=True)
        labels = torch.nn.functional.one_hot(torch.tensor([0, 1]), 10).float()
        loss = ns["classification_criterion"](name)(logits, labels, 0, 10, 10., device=torch.device("cpu"))
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("use_time", [False, True])
def test_optional_evaluation_branches_with_mock_interfaces(notebook_namespace, tmp_path, use_time):
    """Fixture data checks code paths, not scientific BERT or DNA performance."""
    ns = notebook_namespace
    classifier = ns["DNABert"](10)
    regressor = ns["DNABertRegressor"](1, [0.0033433])
    class_path, reg_path = tmp_path / "class.pt", tmp_path / "reg.pt"
    torch.save(classifier.state_dict(), class_path)
    torch.save(regressor.state_dict(), reg_path)
    ns.update(test_data=pd.DataFrame({"sequence": ["AT", "CG", "TA", "GC"],
                                      "class_id": [0, 1, 0, 1], "target": [0., .1, .4, .2]}),
              test_tokenizer=ns["load_tokenizer"](), CLASSIFIER_CHECKPOINT=class_path,
              REGRESSOR_CHECKPOINT=reg_path, SEQUENCE_COLUMN="sequence", CLASS_COLUMN="class_id",
              REGRESSION_COLUMNS=["target"], REGRESSION_TARGETS_TRANSFORMED=False,
              TARGET_SCALER_PATH=None, INFERENCE_BATCH_SIZE=2, RESULTS_DIR=tmp_path,
              PLOT_TIME_COLUMN="collected" if use_time else None, PLOT_TIME_WINDOW=None, plot_times=None)
    if use_time:
        ns["test_data"]["collected"] = ["2024-03-01", "2024-01-01", "2024-01-03", "2024-01-01"]
        ns["plot_times"] = ns["parse_plot_times"](ns["test_data"], "collected")
    for source in code_cells():
        if source.startswith("if test_data is not None and "):
            exec(compile(source, str(PATH), "exec"), ns)
    assert ns["classification_result"]["confusion"].sum() == 4
    assert ns["regression_result"]["mean"].shape == (4, 1)
    for name in ["classification_test.png", "regression_test.png", "classification_predictions.csv", "regression_predictions.csv"]:
        assert (tmp_path / name).stat().st_size > 0
    saved = pd.read_csv(tmp_path / "regression_predictions.csv")
    if use_time:
        np.testing.assert_array_equal(pd.to_datetime(saved["plot_time"]), ns["plot_times"])
        assert ns["axes"][0, 1].get_xlabel() == "collected (source timestamps)"
    else:
        assert ns["axes"][0, 1].get_xlabel() == "Predicted mean (target units)"
    # No lines or filled polygons may bridge independent observations.
    assert not ns["axes"][0, 1].lines
    plt.close("all")

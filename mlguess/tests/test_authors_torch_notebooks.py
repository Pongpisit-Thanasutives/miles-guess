"""Restoration gates and numerical evaluation tests for the authors' Torch examples.

Small arrays here are unit-test fixtures only. The delivered notebooks load the
original Cabauw / precipitation observations; no synthetic example is substituted.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")
from mlguess.torch.models import CategoricalDNN, DNN

ROOT = Path(__file__).resolve().parents[2]


def read_notebook(name):
    return json.loads((ROOT / "notebooks" / name).read_text())


def source(cell):
    return "".join(cell["source"])


def function_from_notebook(name, function):
    notebook = read_notebook(name)
    node = next(node for cell in notebook["cells"] if cell["cell_type"] == "code"
                for node in ast.parse(source(cell)).body
                if isinstance(node, ast.FunctionDef) and node.name == function)
    namespace = {"torch": torch, "np": np, "device": torch.device("cpu")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
    return namespace[function]


@pytest.mark.parametrize("name", ["regression_example_torch.ipynb", "classifier_example_torch.ipynb"])
def test_original_provenance_and_syntax(name):
    notebook = read_notebook(name)
    provenance = notebook["metadata"]["authors_example"]
    original = ROOT / "notebooks" / provenance["source"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == provenance["source_sha256"]
    assert provenance["generated_data"] is False
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(source(cell), name, "exec")
    tags = {tag for cell in notebook["cells"] for tag in cell.get("metadata", {}).get("tags", [])}
    assert {"heldout-evaluation", "heldout-plots", "heldout-calibration-plots"} <= tags


def test_regression_preserves_authors_scaling_and_loader():
    name = "regression_example_torch.ipynb"
    original = json.loads((ROOT / "notebooks/legacy" / name).read_text())
    restored = read_notebook(name)
    by_tag = {tag: cell for cell in restored["cells"] for tag in cell["metadata"].get("tags", [])}
    # Keep the exact original training-only scaler and batch-128 loader cells.
    for index in (11, 12):
        assert source(by_tag[f"authors-cell-{index}"]) == source(original["cells"][index])
    # Split proportions and grouping are scientific choices: compare AST arguments.
    def group_split_choices(cells):
        choices = []
        for cell in cells:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse(source(cell))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GroupShuffleSplit":
                    choices.append({kw.arg: ast.dump(kw.value) for kw in node.keywords})
        return choices
    assert group_split_choices(restored["cells"]) == group_split_choices(original["cells"])


@pytest.mark.parametrize("evidential", [False, True])
def test_regression_batched_evaluation_matches_whole_forward(evidential):
    forward = function_from_notebook("regression_example_torch.ipynb", "batched_forward")
    torch.manual_seed(1000)
    model = DNN(input_size=2, output_size=1, layer_size=[4], dr=0.2,
                batch_norm=False, lng=evidential)
    values = np.array([[0.2, 0.1], [0.7, 0.4], [0.1, 0.5]], dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        expected = model(torch.tensor(values))
    actual = forward(model, values, batch_size=2)
    if evidential:
        for found, reference in zip(actual, expected):
            np.testing.assert_allclose(found, reference.numpy(), rtol=1e-6, atol=1e-7)
    else:
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-6, atol=1e-7)


def test_classification_keeps_original_caps_and_loader():
    name = "classifier_example_torch.ipynb"
    original = json.loads((ROOT / "notebooks/legacy" / name).read_text())
    restored = read_notebook(name)
    by_tag = {tag: cell for cell in restored["cells"] for tag in cell["metadata"].get("tags", [])}
    assert source(by_tag["authors-cell-10"]) == source(original["cells"][10])
    for index in (16, 23):
        def constants(text):
            return {node.targets[0].id: node.value.value for node in ast.parse(text).body
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Constant)}
        before = constants(source(original["cells"][index]))
        after = constants(source(by_tag[f"authors-cell-{index}"]))
        assert after["num_epochs"] == before["num_epochs"] == 100
        assert after["batches_per_epoch"] == before["batches_per_epoch"] == 10


def test_classifier_evaluation_has_correct_probabilities_and_variances():
    predict = function_from_notebook("classifier_example_torch.ipynb", "batched_classifier_predictions")
    # Constant logits distinguish softmax from the Dirichlet mean exactly.
    model = CategoricalDNN(input_size=2, output_size=4, layer_size=[], dr=[],
                           batch_norm=False, softmax=False)
    with torch.no_grad():
        model.fcn[0].weight.zero_()
        model.fcn[0].bias.copy_(torch.tensor([-2., 0., 1., 3.]))
    values = np.array([[1., 0.], [2., 1.], [-1., 2.]], dtype=np.float32)
    standard = predict(model, values, evidential=False, batch_size=2)
    expected_standard = torch.softmax(torch.tensor([-2., 0., 1., 3.]), dim=0).numpy()
    np.testing.assert_allclose(standard, np.tile(expected_standard, (3, 1)), rtol=1e-6)
    probabilities, vacuity, aleatoric, epistemic = predict(model, values, evidential=True, batch_size=2)
    expected = np.array([1., 1., 2., 4.]) / 8
    np.testing.assert_allclose(probabilities, np.tile(expected, (3, 1)))
    np.testing.assert_allclose(vacuity, 0.5)
    np.testing.assert_allclose(aleatoric + epistemic, probabilities * (1-probabilities), rtol=1e-6)
    assert not np.allclose(standard, probabilities)


def test_evidential_training_call_does_not_pass_device_as_class_weights():
    from mlguess.torch.class_losses import edl_digamma_loss
    notebook = read_notebook("classifier_example_torch.ipynb")
    training_cell = next(cell for cell in notebook["cells"]
                         if "authors-cell-23" in cell["metadata"].get("tags", []))
    call = next(node.value for node in ast.parse(source(training_cell)).body
                if isinstance(node, ast.For)
                for nested in node.body if isinstance(nested, ast.For)
                for node in nested.body
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "loss"
                                                       for target in node.targets))
    outputs = torch.tensor([[0.3, -0.2, 1.0, 0.1], [0.4, 1.0, 0.2, -0.3]], requires_grad=True)
    batch_y = torch.tensor([2, 1])
    namespace = {"criterion": edl_digamma_loss, "outputs": outputs,
                 "batch_y_onehot": torch.nn.functional.one_hot(batch_y, 4).float(),
                 "epoch": 5, "num_classes": 4, "batch_y": batch_y}
    loss = eval(compile(ast.Expression(body=call), "notebook loss", "eval"), namespace)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(outputs.grad).all()
    assert torch.count_nonzero(outputs.grad) > 0

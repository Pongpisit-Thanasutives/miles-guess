"""Execute corrected notebook calculations with small independent examples.

These tests validate the edited cells; they do not reproduce historical paper
figures whose datasets and trained model files are external to this package.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import pytest
from scipy.stats import beta, dirichlet


ROOT = Path(__file__).resolve().parents[2]


def notebook(name):
    path = ROOT / "notebooks" / name
    if not path.exists():
        pytest.skip("Notebook sources are not included in this installation")
    result = json.loads(path.read_text(encoding="utf-8"))
    # Keep the historical cell references stable when a notation guide is added.
    result["cells"] = [cell for cell in result["cells"]
                       if not (cell["cell_type"] == "markdown" and
                               "paper-notation-guide" in cell.get("metadata", {}).get("tags", []))]
    return result


def cell_source(name, index):
    return "".join(notebook(name)["cells"][index]["source"])


def ensemble_namespace(monkeypatch):
    # Equally confident members disagree about the class: maxima are always .8,
    # but for class zero its member probabilities are .8 and .1.
    member_probabilities = [np.array([[.8, .1, .05, .05]]),
                            np.array([[.1, .8, .05, .05]])]
    frames = []
    for probability in member_probabilities:
        frame = pd.DataFrame(probability, columns=[f"pred_conf{k+1}" for k in range(4)])
        frame["pred_conf"] = .8
        frame["pred_label"] = np.argmax(probability, axis=1)
        frame["true_label"] = 0
        frame["entropy"] = 0.0
        frame["mutual_info"] = 0.0
        frames.append(frame)

    def read_member(path):
        member = int(Path(path).stem.split("_")[-1])
        return frames[member % 2].copy()

    monkeypatch.setattr(pd, "read_parquet", read_member)
    return {"np": np, "pd": pd, "os": os, "df_det": frames[0].copy(),
            "tqdm": SimpleNamespace(tqdm=lambda values: values)}


def test_paper_categorical_trace_obeys_total_variance(monkeypatch):
    namespace = ensemble_namespace(monkeypatch)
    exec(cell_source("paper_figures_categorical.ipynb", 3), namespace)
    result = namespace["df_ens_stats"]
    np.testing.assert_allclose(namespace["ensemble_mean"], [[.45, .45, .05, .05]])
    assert result["ale_sum"].iloc[0] == pytest.approx(.345)
    assert result["epi_sum"].iloc[0] == pytest.approx(.245)
    assert result["total_sum"].iloc[0] == pytest.approx(.59)


def test_sounding_mixture_uses_one_class_for_all_members(monkeypatch):
    namespace = ensemble_namespace(monkeypatch)
    exec(cell_source("sounding-figure.ipynb", 4), namespace)
    result = namespace["df_det"]
    assert result["pred_label"].iloc[0] == 0
    assert result["pred_conf"].iloc[0] == pytest.approx(.45)
    # Notebook's final presentation columns are standard deviations.
    assert result["aleatoric"].iloc[0] ** 2 == pytest.approx(.125)
    assert result["epistemic"].iloc[0] ** 2 == pytest.approx(.1225)
    assert result["total"].iloc[0] ** 2 == pytest.approx(.2475)


def evidential_namespace():
    probabilities = np.linspace(0, 1, 10001)
    return {"np": np, "dirichlet": dirichlet, "idx": 0,
            "test_preds": np.array([[2.0, 3.0, 4.0, 5.0]]),
            "p_simp": np.column_stack((probabilities, 1 - probabilities))}


def execute_evidential_tail(namespace, index):
    source = cell_source("evidential_figure_1.ipynb", index)
    # The earlier lines retrieve a specific sounding from an external dataset.
    exec(source[source.index("test_ev ="):], namespace)


def test_evidential_figure_distribution_and_moments_use_same_marginal():
    namespace = evidential_namespace()
    execute_evidential_tail(namespace, 11)
    exec(cell_source("evidential_figure_1.ipynb", 12), namespace)
    np.testing.assert_allclose(namespace["marginal_alpha"], [5, 13])
    np.testing.assert_allclose(namespace["dp"], beta.pdf(namespace["p_simp"][:, 0], 5, 13), atol=1e-12)
    expected_epistemic = dirichlet.var(np.array([3., 4., 5., 6.]))[2]
    assert namespace["evi_epi"] == pytest.approx(expected_epistemic)
    mean = 5 / 18
    assert namespace["evi_ale"] + namespace["evi_epi"] == pytest.approx(mean * (1 - mean))


@pytest.mark.parametrize("scaled", [False, True])
def test_evidential_figure_mean_marker_and_density(monkeypatch, scaled):
    namespace = evidential_namespace()
    execute_evidential_tail(namespace, 18 if scaled else 11)
    namespace.update(plt=plt, gridspec=gridspec, Polygon=Polygon,
                     temps=np.linspace(0, -20, 21), dewps=np.linspace(-2, -22, 21),
                     heights=np.arange(0, 5250, 250), us=np.linspace(0, 20, 21),
                     vs=np.linspace(0, 10, 21))
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)
    try:
        exec(cell_source("evidential_figure_1.ipynb", 19 if scaled else 17), namespace)
        namespace["fig"].canvas.draw()
        distribution_axis = namespace["fig"].axes[2 if scaled else 3]
        mean_marker = distribution_axis.patches[-2]
        mean_position = mean_marker.get_x() + mean_marker.get_width() / 2
        factor = namespace["belief_scale"] if scaled else 1.0
        assert mean_position == pytest.approx(factor * 5 / 18)
        curve = distribution_axis.lines[0]
        assert np.trapezoid(curve.get_ydata(), curve.get_xdata()) == pytest.approx(1.0, abs=1e-10)
    finally:
        plt.close(namespace.get("fig"))


@pytest.mark.parametrize("name,first", [
    ("paper_figures_categorical.ipynb", 3),
    ("sounding-figure.ipynb", 4),
    ("evidential_figure_1.ipynb", 11),
])
def test_corrected_notebooks_do_not_keep_stale_scientific_outputs(name, first):
    for cell in notebook(name)["cells"][first:]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


@pytest.mark.parametrize("name", ["classifier_example.ipynb", "regression_example.ipynb"])
def test_documentation_notebook_copies_match(name):
    document = ROOT / "docs" / "source" / "notebooks" / name
    if not document.exists():
        pytest.skip("Documentation notebook copies are not installed")
    assert notebook(name) == json.loads(document.read_text(encoding="utf-8"))

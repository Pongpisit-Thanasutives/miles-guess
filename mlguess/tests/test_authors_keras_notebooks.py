"""Scientific evaluation checks for the retained authors' Keras examples."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]


def notebook_source(name):
    notebook = json.loads((ROOT / 'notebooks' / name).read_text())
    return '\n\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')


def regression_evaluator():
    source = ast.parse(notebook_source('regression_example.ipynb'))
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'point_metrics', 'evaluate_gaussian_mixture'}]
    namespace = dict(np=np, pd=pd, norm=norm, logsumexp=logsumexp, output_cols=['friction_velocity'])
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook evaluation>', 'exec'), namespace)
    return namespace['evaluate_gaussian_mixture']


def test_single_gaussian_notebook_evaluation_matches_distribution():
    evaluate = regression_evaluator()
    observed = np.array([[0.], [1.], [2.]])
    means = np.array([[[.1], [.8], [2.2]]])
    variances = np.array([[[.25], [.16], [.36]]])
    result = evaluate(observed, means, variances)
    distribution = norm(loc=means[0], scale=np.sqrt(variances[0]))
    np.testing.assert_allclose(result['intervals'][.9][0], distribution.ppf(.05))
    np.testing.assert_allclose(result['intervals'][.9][1], distribution.ppf(.95))
    np.testing.assert_allclose(result['metrics']['nll'], -distribution.logpdf(observed).mean(axis=0))
    np.testing.assert_allclose(result['aleatoric'], variances[0])
    np.testing.assert_allclose(result['epistemic'], 0)


def test_mixture_notebook_quantiles_and_variance_are_actual_mixture():
    evaluate = regression_evaluator()
    means = np.array([[[-2.], [1.]], [[3.], [4.]]])
    variances = np.array([[[.25], [1.]], [[2.], [.16]]])
    result = evaluate(np.array([[0.], [2.]]), means, variances)
    for level, (lower, upper) in result['intervals'].items():
        np.testing.assert_allclose(norm.cdf(lower[None], means, np.sqrt(variances)).mean(axis=0), (1-level)/2, atol=1e-13)
        np.testing.assert_allclose(norm.cdf(upper[None], means, np.sqrt(variances)).mean(axis=0), (1+level)/2, atol=1e-13)
    expected = np.mean(variances + means**2, axis=0) - means.mean(axis=0)**2
    np.testing.assert_allclose(result['total'], expected)
    np.testing.assert_allclose(result['epistemic'], means.var(axis=0, ddof=0))


def test_authors_examples_keep_data_models_and_heldout_evaluation():
    regression = notebook_source('regression_example.ipynb')
    classifier = notebook_source('classifier_example.ipynb')
    for source in (regression, classifier):
        assert 'make_regression' not in source and 'make_classification' not in source
        ast.parse(source)
    assert 'sample_cabauw_surface_layer.csv' in regression
    assert all(path in regression for path in ('surface_layer/mlp.yml', 'surface_layer/gaussian.yml', 'surface_layer/evidential.yml'))
    assert 'member_scalers=member_y_scalers' in regression
    assert 'member_input_scalers=member_x_scalers' in regression
    assert 'ptype/deterministic.yml' in classifier and 'ptype/evidential.yml' in classifier
    assert 'mlp = CategoricalDNN(**conf["model"],callbacks=[])' in classifier
    assert 'scaler_type=conf["scaler_type"],encoder_type="onehot",groups=[]' in classifier
    assert 'true_labels = np.argmax(data["test"]' in classifier
    assert 'mlp.predict(x_test)' in classifier
    assert 'ev_mlp.predict(x_test,return_uncertainties=True' in classifier
    assert 'ev_mlp = EvidentialCategoricalDNN(**ev_options,callbacks=[])' in classifier

"""Construct unfitted preprocessing transforms from configuration."""

from collections.abc import Mapping
import logging

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer

logger = logging.getLogger(__name__)


def load_preprocessor(stype, params=None, seed=1000):
    """Return an unfitted scaler, honoring explicitly supplied constructor options.

    Quantile transforms use ``seed`` unless ``random_state`` is supplied in
    ``params``. Unsupported options raise an informative error rather than being
    silently discarded. Fit the returned object on training data only.
    """
    if params is not None and not isinstance(params, Mapping):
        raise TypeError("Scaler params must be a mapping or None")
    choices = {
        "standard": (StandardScaler, {}),
        "normalize": (MinMaxScaler, {"feature_range": (0, 1)}),
        "symmetric": (MinMaxScaler, {"feature_range": (-1, 1)}),
        "robust": (RobustScaler, {}),
        "quantile": (QuantileTransformer, {"n_quantiles": 1000, "random_state": seed, "output_distribution": "normal"}),
        "quantile-uniform": (QuantileTransformer, {"n_quantiles": 1000, "random_state": seed, "output_distribution": "uniform"}),
    }
    if stype not in choices:
        raise ValueError(f"Unknown preprocessing type {stype!r}; choose from {', '.join(choices)}")
    constructor, defaults = choices[stype]
    options = {**defaults, **({} if params is None else params)}
    if "feature_range" in options:
        options["feature_range"] = tuple(options["feature_range"])
    try:
        return constructor(**options)
    except TypeError as exc:
        raise TypeError(f"Invalid parameters for {stype!r} preprocessing: {exc}") from exc


def load_preprocessing(conf, seed=1000):
    """Construct input/target scalers without fitting either on any split."""
    scalers = []
    for name in ("scaler_x", "scaler_y"):
        spec = conf["data"].get(name)
        if spec is None or spec is False:
            logger.info("Loading %s: None", name)
            scalers.append(False)
        else:
            logger.info("Loading %s: %s", name, spec["type"])
            scalers.append(load_preprocessor(spec["type"], spec.get("params", {}), seed))
    return tuple(scalers)

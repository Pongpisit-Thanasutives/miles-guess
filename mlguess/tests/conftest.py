"""Backend-independent tests run without optional Keras/TensorFlow installs."""
import importlib.util
import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("MPLBACKEND", "Agg")

collect_ignore = []
if importlib.util.find_spec("keras") is None:
    collect_ignore += ["test_layers.py", "test_models.py", "test_keras_lotv.py", "test_legacy_lotv.py"]
if importlib.util.find_spec("torch") is None:
    collect_ignore += [
        "test_torch_model.py", "test_torch_checkpoint.py", "test_modern_core.py",
        "test_modern_trainers.py", "test_scientific_regression.py", "test_applications.py", "test_torch_lotv.py",
    ]

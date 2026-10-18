"""Exercise CSV classification and multi-target regression using generated test fixtures.

Run from the repository root: python examples/validate_custom_notebook.py.
Temporary fixtures and model outputs are removed when validation finishes.
These are automated tests, not datasets or results for the authors' examples.
"""
from pathlib import Path
import tempfile

import nbformat
import numpy as np
import pandas as pd

from execute_notebooks import execute_notebook


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/evidential_learning_your_data.ipynb"
    with tempfile.TemporaryDirectory(prefix="miles-guess-csv-") as directory:
        temporary = Path(directory)
        rng = np.random.default_rng(14)
        # 109 rows leave a final singleton training minibatch; do not drop it.
        frame = pd.DataFrame(rng.normal(size=(109, 3)), columns=["x1", "x2", "x3"])
        frame["y1"] = frame.x1 + rng.normal(0, .1, len(frame))
        frame["y2"] = -2*frame.x2 + rng.normal(0, .2, len(frame))
        frame["label"] = np.asarray(["low", "medium", "high"])[np.arange(len(frame)) % 3]
        csv = temporary / "own_data.csv"
        frame.to_csv(csv, index=False)
        for task, targets in [("regression", ["y1", "y2"]), ("classification", ["label"])]:
            notebook = nbformat.read(source, as_version=4)
            configuration = notebook.cells[2].source
            configuration += (f"\nTASK = {task!r}\nDATA_PATH = Path({str(csv)!r})\n"
                              f"FEATURE_COLUMNS = ['x1', 'x2', 'x3']\nTARGET_COLUMNS = {targets!r}\n"
                              f"EPOCHS = 4\nBATCH_SIZE = 64\nSPLIT_STRATEGY = 'random'\nTEST_FRACTION = 0.2\nVALIDATION_FRACTION = 0.2\nOUTPUT_DIR = Path({str(temporary / task)!r})\n"
                              "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n")
            notebook.cells[2].source = configuration
            notebook.cells.append(nbformat.v4.new_code_cell(
                "assert len(train_idx) == 65\n"
                "assert len(prediction_table) == len(test_idx) == 22\n"
                "assert np.isfinite(np.asarray(list(evaluation['metrics'].values()))).all() "
                "if TASK == 'classification' else np.isfinite(evaluation['metrics'].to_numpy()).all()\n"
                "assert (OUTPUT_DIR/'model.pt').exists()\n"))
            path = temporary / f"custom_csv_{task}.ipynb"
            nbformat.write(notebook, path)
            execute_notebook(path, cwd=root, write=False)


if __name__ == "__main__":
    main()

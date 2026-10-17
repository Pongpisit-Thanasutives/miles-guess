"""Execute notebook cells in order with IPython, without opening kernel sockets.

Usage::

    python examples/execute_notebooks.py notebooks/regression_example.ipynb
    python examples/execute_notebooks.py notebooks/a.ipynb notebooks/b.ipynb \
        --figures-dir validation/notebook_figures

Each notebook gets a fresh IPython namespace and runs from its own directory by
default. All code cells must succeed before the executed notebook is saved. This
is useful in restricted environments where nbclient cannot start a local kernel;
ordinary users can also use Jupyter's Restart Kernel and Run All. Only execute
notebooks whose code you trust. Requires the package's ``notebooks`` extra.
"""
from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.core.pylabtools import select_figure_formats
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support, flush_figures


def execute_notebook(path, *, cwd=None, write=True, figures_dir=None):
    """Run a notebook and return its execution summary; raise on the first error.

    Parameters are paths. ``cwd=None`` uses the notebook's parent directory.
    ``write=False`` validates cells without overwriting the source notebook.
    ``figures_dir`` optionally exports captured PNG/SVG figures with stable names.
    """
    path = Path(path).resolve(strict=True)
    workdir = path.parent if cwd is None else Path(cwd).resolve(strict=True)
    figures_dir = None if figures_dir is None else Path(figures_dir).resolve()
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    InteractiveShell.clear_instance()
    shell = InteractiveShell.instance()
    configure_inline_support(shell, "inline")
    # Inline backend setup is cached globally; a fresh shell needs its own formatter.
    select_figure_formats(shell, {"png"})
    count = 0
    print(f"Executing {path.name}", flush=True)
    try:
        os.chdir(workdir)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output(display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
                flush_figures()
            plt.close("all")
            if not result.success:
                print(captured.stdout, end="", flush=True)
                print(captured.stderr, end="", file=sys.stderr, flush=True)
                error = result.error_in_exec or result.error_before_exec
                raise RuntimeError(f"{path.name}: code cell {index} failed") from error
            cell.execution_count = count
            cell.outputs = []
            for channel in ("stdout", "stderr"):
                stream = getattr(captured, channel)
                if stream:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=channel, text=stream))
            for item in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=item.data,
                                                          metadata=item.metadata))
            print(f"  code cell {index}: passed", flush=True)
    finally:
        os.chdir(previous_cwd)
        plt.close("all")
        InteractiveShell.clear_instance()

    elapsed = time.perf_counter() - started
    notebook.metadata["execution"] = {
        "method": "IPython in-process; all code cells in order",
        "python": sys.version.split()[0],
        "runtime_seconds": round(elapsed, 3),
    }
    notebook.metadata.setdefault("language_info", {})["version"] = sys.version.split()[0]
    nbformat.validate(notebook)
    if write:
        nbformat.write(notebook, path)

    figure_count = 0
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" not in data and "image/svg+xml" not in data:
                continue
            figure_count += 1
            if figures_dir is not None:
                figures_dir.mkdir(parents=True, exist_ok=True)
                basename = f"{path.stem}_{figure_count:02d}"
                if "image/png" in data:
                    (figures_dir / f"{basename}.png").write_bytes(base64.b64decode(data["image/png"]))
                if "image/svg+xml" in data:
                    (figures_dir / f"{basename}.svg").write_text(data["image/svg+xml"], encoding="utf-8")
    summary = dict(notebook=str(path), code_cells=count, figures=figure_count,
                   runtime_seconds=round(elapsed, 3), saved=write)
    print(f"Completed {path.name}: {count} code cells, {figure_count} figures, {elapsed:.1f}s", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path, help="Notebook paths to execute sequentially")
    parser.add_argument("--cwd", type=Path, help="Working directory for every notebook (default: its own folder)")
    parser.add_argument("--figures-dir", type=Path, help="Optional directory for captured PNG/SVG figures")
    parser.add_argument("--validation-only", action="store_true", help="Run all cells without overwriting notebooks")
    args = parser.parse_args(argv)
    paths = [path.resolve(strict=True) for path in args.notebooks]
    for path in paths:
        execute_notebook(path, cwd=args.cwd, write=not args.validation_only, figures_dir=args.figures_dir)


if __name__ == "__main__":
    main()

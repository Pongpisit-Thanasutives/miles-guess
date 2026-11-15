"""Execute all teaching notebooks on a temporary schema fixture (never on research data).

Run from the project root, in its installed environment:
    python tests/run_notebooks.py

Optional --work selects an empty directory in which to retain study outputs
and executed notebook copies. No outputs are written into the source notebooks.
"""
import argparse
import subprocess
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from schema_fixture import write_fixture


def run(work, in_process=False):
    import nbformat
    from nbclient import NotebookClient

    root = Path(__file__).resolve().parents[1]
    work = Path(work).resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError(f"Choose an empty QA directory: {work}")
    work.mkdir(parents=True, exist_ok=True)
    splits = write_fixture(work / "schema-mock")
    env = os.environ.copy()
    env.update(PM25_DATA_DIR=str(splits), PM25_RUN_DIR=str(work / "runs"),
               PM25_FEATURE_OPTION="at-bb-aod", PM25_N_TRIALS="1",
               PM25_MAX_EPOCHS="2", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # Register this Python only for this check, without modifying the user's kernels.
    kernel_name = "pm25-edl-qa"
    specs = work / "jupyter"
    kernel = specs / "kernels" / kernel_name
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "PM2.5 EDL QA", "language": "python"}))
    old = os.environ.get("JUPYTER_PATH")
    os.environ["JUPYTER_PATH"] = str(specs) + (os.pathsep + old if old else "")
    try:
        summary = []
        for phase in ("hpt", "usage"):
            books = sorted((root / "notebooks" / phase).glob("*.ipynb"))
            if len(books) != 6:
                raise RuntimeError(f"Expected six {phase} notebooks, found {len(books)}")
            output = work / phase
            output.mkdir()
            for path in books:
                print(f"Executing {phase}/{path.name}", flush=True)
                nb = nbformat.read(path, as_version=4)
                nbformat.validate(nb)
                if in_process:
                    # Offline runner for restricted hosts. Every notebook still gets a fresh
                    # interpreter and an IPython kernel; no network transport is exercised.
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--single",
                                    str(path), str(output / path.name)], check=True,
                                   env=env, cwd=path.parent)
                    nb = nbformat.read(output / path.name, as_version=4)
                else:
                    client = NotebookClient(nb, kernel_name=kernel_name, timeout=180,
                                            resources={"metadata": {"path": str(path.parent)}})
                    client.execute(env=env)
                code = [c for c in nb.cells if c.cell_type == "code"]
                assert all(c.execution_count is not None for c in code)
                images = sum("image/png" in out.get("data", {}) for c in code for out in c.outputs)
                if images < (3 if phase == "usage" else 1):
                    raise AssertionError(f"Missing inline plotting outputs in {path.name}: {images}")
                summary.append(dict(notebook=f"{phase}/{path.name}", code_cells=len(code), inline_pngs=images))
                nbformat.write(nb, output / path.name)
        (work / "execution.json").write_text(json.dumps(summary, indent=2))
    finally:
        if old is None:
            os.environ.pop("JUPYTER_PATH", None)
        else:
            os.environ["JUPYTER_PATH"] = old


def execute_inprocess(path, output):
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    import base64
    import nbformat
    from ipykernel.inprocess.manager import InProcessKernelManager

    nb = nbformat.read(path, as_version=4)
    kernel = InProcessKernelManager()
    kernel.start_kernel()
    shell = kernel.kernel.shell
    # A normal interactive notebook uses the inline backend. In-process kernels
    # do not select it automatically on a headless test host.
    shell.run_line_magic("matplotlib", "inline")
    nb.metadata["qa_execution"] = "Fresh Python, in-process IPython kernel (no ZeroMQ transport)"
    try:
        for c in nb.cells:
            if c.cell_type != "code":
                continue
            outputs = []
            def capture(data, metadata=None, **kwargs):
                data = {k: (base64.b64encode(v).decode() if isinstance(v, bytes) else v)
                        for k, v in data.items()}
                outputs.append(nbformat.v4.new_output("display_data", data=data, metadata=metadata or {}))
            shell.display_pub.publish = capture
            stdout, stderr = StringIO(), StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = shell.run_cell(c.source, store_history=True)
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                print(stdout.getvalue(), stderr.getvalue(), file=sys.stderr)
                raise RuntimeError(f"{path.name}, cell {c.get('id')}: {error}") from error
            c.execution_count = result.execution_count
            c.outputs = [nbformat.v4.new_output("stream", name=k, text=v.getvalue())
                         for k, v in [("stdout", stdout), ("stderr", stderr)] if v.getvalue()] + outputs
    finally:
        kernel.shutdown_kernel()
    nbformat.write(nb, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, help="Empty directory for the temporary mock experiment")
    parser.add_argument("--in-process", action="store_true", help="Use fresh in-process IPython kernels on hosts that forbid Jupyter sockets")
    parser.add_argument("--single", nargs=2, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single:
        execute_inprocess(*args.single)
    elif args.work:
        run(args.work, args.in_process)
    else:
        with TemporaryDirectory(prefix="pm25-notebook-check-") as d:
            run(d, args.in_process)

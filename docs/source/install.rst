Install MILES GUESS
===================

This source distribution targets Python 3.11 and NumPy 2. The audited CPU
runtime uses Python 3.11.17, PyTorch 2.14.1, and NumPy 2.4.6. Native PyTorch
requires no Keras or TensorFlow installation. Install this local source
package to obtain the modernization; the public PyPI package is separate.

Conda and CPU PyTorch
--------------------

From a Conda-enabled terminal, create a Python environment, activate it, and
enter the unpacked project directory::

    conda create -n miles_guess_cpu python=3.11 pip -y
    conda activate miles_guess_cpu
    cd /path/to/miles-guess
    python -m pip install --upgrade pip

On Windows x86-64 or Linux, install the CPU-specific wheel::

    python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

On macOS 14 or later with native Apple Silicon Python, install the PyPI wheel::

    python -m pip install torch==2.14.1

Official CPython 3.11 wheels are listed in the
`PyTorch CPU index <https://download.pytorch.org/whl/cpu/torch/>`_.
Intel macOS and Rosetta Python are outside this installation target.

Then install the local library and notebook tools::

    python -m pip install -c requirements/validated-py311.txt -e ".[torch,notebooks]"
    python -m pip check
    python examples/cpu_quickstart.py
    python examples/uncertainty_quickstart.py
    python -m jupyter lab

The notebook entry guide is ``notebooks/README.md``. Use ``.[torch]`` instead
if notebook tools are unnecessary. The constraints file pins packages used
in the audit; it does not install optional extras or lock every transitive
package. The Jupyter interface packages were not part of the validation
runtime and are not pinned.

``environment_torch.yml`` creates the same Python/pip bootstrap environment.
Continue with the platform-specific pip commands above. The top-level README
also provides virtual-environment instructions without Conda.

Verification and optional APIs
------------------------------

Install the test tools and run the suite::

    python -m pip install -c requirements/validated-py311.txt -e ".[torch,test]"
    python -m pytest -q

For Keras and data coverage, install ``.[torch,keras,data,test]`` using the
same constraints and set ``KERAS_BACKEND=torch`` before importing Keras or
running pytest. In PowerShell use ``$env:KERAS_BACKEND="torch"``; in bash/zsh
use ``export KERAS_BACKEND=torch``. Tests for unavailable extras skip.
TensorFlow and JAX, geospatial packages, tuning services, and documentation
tools are separate optional extras.

Package compatibility remains ``torch>=2.5.1,<3`` and ``numpy>=2,<3``. The
validated constraint set selects PyTorch 2.14.1 and NumPy 2.4.6. Windows,
Apple Silicon macOS, and Linux CI jobs are configured with validated and
latest-compatible dependencies; those native CI jobs were not executed for
this delivery. See ``VALIDATION.md`` for the actual test record and limits.

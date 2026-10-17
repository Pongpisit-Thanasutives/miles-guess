"""Optional PBS script generation; submission requires a working local qsub."""

import getpass
import logging
import os
from pathlib import Path
import shlex
import subprocess

import yaml


def _read_job_config(config_file):
    with open(config_file, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not config.get("pbs"):
        raise ValueError("The configuration must include PBS options; see config/pbs_example.yaml")
    save_path = Path(os.path.expandvars(os.path.expanduser(config["save_loc"]))).resolve()
    save_path.mkdir(parents=True, exist_ok=True)
    return config, save_path


def _submit_or_return(path, launch):
    if not launch:
        return str(path)
    result = subprocess.run(["qsub", str(path)], check=True, capture_output=True, text=True)
    job_id = result.stdout.strip()
    logging.info("Launched PBS job %s using %s", job_id, path)
    return job_id


def launch_pbs_jobs(config_file, trainer_path, args="", launch=True):
    """Generate a single-job PBS script and optionally submit it.

    Set ``launch=False`` to inspect the script without submitting anything.
    The generated script is retained in ``save_loc``. Additional arguments can
    be a shell-like string or a sequence; they are quoted as individual values.
    Submission failures are raised rather than reported as successful jobs.
    """
    config, save_path = _read_job_config(config_file)
    options = config["pbs"]
    arguments = shlex.split(args) if isinstance(args, str) else list(args)
    command = shlex.join(["python", str(Path(trainer_path).resolve()), "-c", str(Path(config_file).resolve()), *map(str, arguments)])
    lines = [
        "#!/bin/bash -l",
        f"#PBS -N {options['name']}",
        f"#PBS -l select={options['select']}:ncpus={options['ncpus']}:ngpus={options['ngpus']}:mem={options['mem']}",
        f"#PBS -l walltime={options['walltime']}",
    ]
    for name in ("gpu_type", "cpu_type"):
        if name in options:
            lines.append(f"#PBS -l {name}={options[name]}")
    lines.extend([
        f"#PBS -A {options['account']}", f"#PBS -q {options['queue']}",
        f"#PBS -o {shlex.quote(str(save_path / 'out'))}",
        f"#PBS -e {shlex.quote(str(save_path / 'out'))}",
        options.get("env_setup", ""), command,
    ])
    path = save_path / "launch_pbs.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _submit_or_return(path, launch)


def launch_distributed_jobs(config_file, script_path, launch=True):
    """Generate the optional Derecho MPI/PBS launcher, or submit when requested.

    This launcher is cluster-specific and requires PBS, MPI, CUDA, module, and
    conda commands on that cluster. ``launch=False`` performs local generation
    only. CUDA ordinals are local to each node, not the total across all nodes.
    Multi-rank CSV exporting in the bundled applications remains unsupported.
    """
    config, save_path = _read_job_config(config_file)
    options = config["pbs"]
    nodes = int(options.get("nodes", 1))
    gpus = int(options.get("ngpus", 1))
    if nodes < 1 or gpus < 1:
        raise ValueError("Distributed PBS nodes and ngpus must both be positive")
    user = getpass.getuser()
    config_path = save_path / "model.yml"
    # Serialize the loaded object: config_file may itself be model.yml.
    with config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    visible_devices = ",".join(map(str, range(gpus)))
    trainer = shlex.quote(str(Path(script_path).resolve()))
    configuration = shlex.quote(str(config_path))
    conda_env = shlex.quote(str(options.get("conda", "holodec")))
    scratch = shlex.quote(f"/glade/derecho/scratch/{user}/")
    lines = [
        "#!/bin/bash",
        f"#PBS -A {options.get('project', 'default_project')}",
        f"#PBS -N {options.get('job_name', 'default_job')}",
        f"#PBS -l walltime={options.get('walltime', '00:10:00')}",
        f"#PBS -l select={nodes}:ncpus={options.get('ncpus', 1)}:ngpus={gpus}:mem={options.get('mem', '4GB')}",
        f"#PBS -q {options.get('queue', 'default_queue')}",
        "#PBS -j oe", "#PBS -k eod", "", "module purge",
        "module load nvhpc cuda cray-mpich conda", f"conda activate {conda_env}", "",
        'nodes=( $( cat "$PBS_NODEFILE" ) )', 'head_node="${nodes[0]}"',
        '''head_node_ip=$(ssh "$head_node" hostname -i | awk '{print $1}')''',
        f"export LSCRATCH={scratch}", "export LOGLEVEL=INFO", "export NCCL_DEBUG=INFO",
        f'CUDA_VISIBLE_DEVICES="{visible_devices}" mpiexec -n {nodes} --ppn 1 --cpu-bind none '
        f'torchrun --nnodes={nodes} --nproc-per-node={gpus} --rdzv-backend=c10d '
        f'--rdzv-endpoint="$head_node_ip" {trainer} -c {configuration}',
    ]
    path = save_path / "launch.sh"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _submit_or_return(path, launch)

#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path
from string import Template

# Define the SLURM job template.
TEMPLATE_JOB = r"""#!/bin/bash -l
#SBATCH --gres=gpu:a100:${n_gpus}
#SBATCH --time=${hours}:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${n_cpus}
#SBATCH --export=NONE
#SBATCH --output=${output_file}
#SBATCH --job-name=${job_name}

unset SLURM_EXPORT_ENV
module load python
conda activate lpc

while :
do
MASTER_PORT=$(shuf -i 20000-65000 -n 1)
if ! lsof -i:$MASTER_PORT -t >/dev/null; then
    break
fi
done

echo "MASTER_PORT: $MASTER_PORT"
echo "MASTER_ADDR: $SLURM_LAUNCH_NODE_IPADDR"

srun -N "$SLURM_JOB_NUM_NODES" --cpu-bind=verbose \
  /bin/bash -c "torchrun --nnodes=\$SLURM_JOB_NUM_NODES --nproc-per-node=\$SLURM_GPUS_ON_NODE --master-addr=\$SLURM_LAUNCH_NODE_IPADDR --master-port=\"$MASTER_PORT\" --start-method=forkserver --node-rank=\$SLURM_NODEID  ${model_script} --config  ${config}  --lr ${lr} --store-penultimate ${store_penultimate} --results-dir ${results_dir}/${id_name}/${k_dir}  --dataset-dir ${dataset_dir} --sample ${i_lr}"
"""

def create_replace_directory(dir_path: Path) -> None:
    """
    Create a new directory or prompt to replace it if it already exists.
    """
    if dir_path.exists():
        response = input(f"The directory {dir_path} already exists. Remove and create a new one? (Y/N): ")
        if response.strip().lower() == 'y':
            response_confirm = input(
                f"Are you sure you want to remove the directory {dir_path}? This action cannot be undone. (Y/N): "
            )
            if response_confirm.strip().lower() == 'y':
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Removed and recreated the directory: {dir_path}")
            else:
                print(f"Action canceled. Using the existing directory: {dir_path}")
        else:
            print(f"Using the existing directory: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created the directory: {dir_path}")

def submit_job_script(script_content: str, job_filename: Path) -> None:
    """
    Write the job script to a file, submit it via sbatch, and then remove the file.
    """
    job_filename.write_text(script_content)
    try:
        subprocess.run(["sbatch", str(job_filename)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job script {job_filename}: {e}")
    finally:
        job_filename.unlink()  # Remove the file after submission

def submit_slurm_jobs(template: str, config: str, id_name: str, dataset_dir: str,
                      results_dir: str, output_dir: str, store_penultimate: str,
                      start_lr: float, factor_lr: float, steps_lr: int,
                      hours: int, n_gpus: int, n_cpus: int) -> None:
    """
    Generate and submit SLURM job scripts based on the provided template and parameters.
    """
    # Set up the results and output directories using pathlib.
    results_path = Path(results_dir) / id_name
    output_path = Path(output_dir) / id_name
    create_replace_directory(results_path)
    create_replace_directory(output_path)

    # Determine the directory of this script.
    script_path = Path(__file__).resolve().parent

    # Create a Template instance from the SLURM job template.
    job_template = Template(template)

    # Define job configurations for each architecture variant.
    job_variants = [
        ("lpc", f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True"),
        ("lpc_wide", f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True --penultimate-nodes wide"),
        ("lpc_narrow", f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True --penultimate-nodes narrow"),
        ("lpc_scl", f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True --scl True"),
        ("lpc_no_pen", f"{script_path / 'main.py'} --architecture-type no_pen --l2-loss True"),
        ("no_pen", f"{script_path / 'main.py'} --architecture-type no_pen"),
        ("no_pen_scl", f"{script_path / 'main.py'} --architecture-type no_pen --scl True"),
        ("no_pen_arcface", f"{script_path / 'main.py'} --architecture-type no_pen --arcface True"),
        ("no_pen_cosface", f"{script_path / 'main.py'} --architecture-type no_pen --cosface True"),
        ("lin_pen", f"{script_path / 'main.py'} --architecture-type lin_pen"),
        ("nonlin_pen", f"{script_path / 'main.py'} --architecture-type nonlin_pen")
    ]

    for k in [1,2,3,4,5]:
        current_results_dir = results_path / str(k)
        current_output_dir = output_path / str(k)
        current_results_dir.mkdir(parents=True, exist_ok=True)
        current_output_dir.mkdir(parents=True, exist_ok=True)

        # Copy the config file into the current results directory.
        shutil.copy2(config, current_results_dir / Path(config).name)

        lr = start_lr
        for i in range(steps_lr + 1):
            for architecture_name, model_script in job_variants:
                output_file = current_output_dir / f"{architecture_name}_{i}.out"
                job_name = f"{id_name}_{k}_{i}_{architecture_name}"

                # Prepare the dictionary of substitutions.
                substitutions = {
                    "n_gpus": str(n_gpus),
                    "n_cpus": str(n_cpus),
                    "hours": str(hours),
                    "output_file": str(output_file),
                    "job_name": job_name,
                    "config": config,
                    "lr": str(lr),
                    "results_dir": results_dir,
                    "id_name": id_name,
                    "dataset_dir": dataset_dir,
                    "store_penultimate": store_penultimate,
                    "k_dir": str(k),
                    "i_lr": str(i),
                    "model_script": model_script
                }

                # Substitute the values into the job template.
                job_script = job_template.safe_substitute(substitutions)
                script_filename = Path(f"run_job_{architecture_name}_{i}_{k}.sh")
                submit_job_script(job_script, script_filename)
            lr *= factor_lr

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate and submit SLURM job scripts.")
    parser.add_argument('--hours', type=int, required=True, help="Maximum time (in hours) to finish the job")
    parser.add_argument('--n-gpus', type=int, required=True, help="Number of GPUs")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--id-name', type=str, required=True, help="Experiment ID name")
    parser.add_argument('--dataset-dir', type=str, required=True, help="Dataset directory")
    parser.add_argument('--results-dir', type=str, required=True, help="Results directory")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory")
    parser.add_argument('--store-penultimate', action='store_true', help="Flag to store penultimate")
    return parser.parse_args()

def main():
    args = parse_args()

    store_penultimate = str(args.store_penultimate)
    
    submit_slurm_jobs(
        template=TEMPLATE_JOB,
        config=args.config,
        id_name=args.id_name,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        store_penultimate=store_penultimate,
        start_lr=0.0001,
        factor_lr=2,
        steps_lr=4,
        hours=args.hours,
        n_gpus=args.n_gpus,
        n_cpus=16 * args.n_gpus
    )

if __name__ == '__main__':
    main()

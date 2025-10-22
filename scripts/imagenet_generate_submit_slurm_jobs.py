#!/usr/bin/env python3
"""
Generate and submit the ImageNet SLURM jobs used in the LPC paper.
Supports both the full sweep (matching the publication) and targeted single-job submissions.
"""
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

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

def create_or_use_directory(dir_path: Path, skip_prompt: bool = False) -> None:
    """
    Create a new directory or use existing one.
    If skip_prompt is True, always use existing directory without prompting.
    """
    if dir_path.exists():
        if not skip_prompt:
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
        print(f"Successfully submitted job: {job_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job script {job_filename}: {e}")
    finally:
        job_filename.unlink()  # Remove the file after submission

def get_model_script_for_variant(script_path: Path, variant: str) -> str:
    """
    Get the model script command for a given variant.
    """
    variant_configs = {
        "lpc": f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True",
        "lpc_wide": f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True --penultimate-nodes wide",
        "lpc_narrow": f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True --penultimate-nodes narrow",
        "no_pen": f"{script_path / 'main.py'} --architecture-type no_pen",
    }
    
    if variant not in variant_configs:
        raise ValueError(f"Unknown variant: {variant}. Available variants: {list(variant_configs.keys())}")
    
    return variant_configs[variant]

def submit_single_job(args):
    """
    Submit a single SLURM job with specific k, variant, and learning rate.
    """
    # Set up paths
    results_path = Path(args.results_dir) / args.id_name
    output_path = Path(args.output_dir) / args.id_name
    
    # Create directories if needed (skip prompt when called from bash script)
    create_or_use_directory(results_path, skip_prompt=args.skip_directory_prompt)
    create_or_use_directory(output_path, skip_prompt=args.skip_directory_prompt)
    
    # Determine the directory of this script
    script_path = Path(__file__).resolve().parent
    
    # Create a Template instance from the SLURM job template
    job_template = Template(TEMPLATE_JOB)
    
    # Get the config file name (without path)
    config_filename = Path(args.config).name
    
    # Set up k-specific directories
    current_results_dir = results_path / str(args.k)
    current_output_dir = output_path / str(args.k)
    current_results_dir.mkdir(parents=True, exist_ok=True)
    current_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy the config file into the current results directory
    copied_config_path = current_results_dir / config_filename
    shutil.copy2(args.config, copied_config_path)
    config_to_use = str(copied_config_path)
    
    # Get the model script for the variant
    model_script = get_model_script_for_variant(script_path, args.job_variant)
    
    # Set up output file and job name
    output_file = current_output_dir / f"{args.job_variant}_{args.lr_label}.out"
    job_name = f"{args.id_name}_{args.k}_{args.lr_label}_{args.job_variant}"
    
    # Prepare the dictionary of substitutions
    substitutions = {
        "n_gpus": str(args.n_gpus),
        "n_cpus": str(16 * args.n_gpus),
        "hours": str(args.hours),
        "output_file": str(output_file),
        "job_name": job_name,
        "config": config_to_use,
        "lr": str(args.lr),
        "results_dir": args.results_dir,
        "id_name": args.id_name,
        "dataset_dir": args.dataset_dir,
        "store_penultimate": str(args.store_penultimate),
        "k_dir": str(args.k),
        "i_lr": str(args.lr_label),
        "model_script": model_script
    }
    
    # Substitute the values into the job template
    job_script = job_template.safe_substitute(substitutions)
    script_filename = Path(f"run_job_{args.job_variant}_{args.lr_label}_{args.k}.sh")
    
    print(f"\nSubmitting job: {job_name}")
    print(f"  Variant: {args.job_variant}")
    print(f"  K value: {args.k}")
    print(f"  Learning rate: {args.lr} (label: {args.lr_label})")
    
    submit_job_script(job_script, script_filename)

def submit_all_jobs(args):
    """
    Submit all jobs based on the original behavior (for backward compatibility).
    """
    # Set up the results and output directories using pathlib
    results_path = Path(args.results_dir) / args.id_name
    output_path = Path(args.output_dir) / args.id_name
    create_or_use_directory(results_path, skip_prompt=args.skip_directory_prompt)
    create_or_use_directory(output_path, skip_prompt=args.skip_directory_prompt)
    
    # Determine the directory of this script
    script_path = Path(__file__).resolve().parent
    
    # Create a Template instance from the SLURM job template
    job_template = Template(TEMPLATE_JOB)
    
    # Get the config file name (without path)
    config_filename = Path(args.config).name
    
    # Define job configurations for each architecture variant
    # Only include lpc and no_pen as per requirements
    job_variants = [
        ("lpc", f"{script_path / 'main.py'} --architecture-type lin_pen --l2-loss True"),
        ("no_pen", f"{script_path / 'main.py'} --architecture-type no_pen"),
    ]
    
    # Define learning rates to use
    learning_rates = [
        (0, 0.0001),
        (1, 0.001),
        (2, 5e-05),
    ]
    
    # K values
    k_values = [1, 2, 3, 4, 5]
    
    for k in k_values:
        current_results_dir = results_path / str(k)
        current_output_dir = output_path / str(k)
        current_results_dir.mkdir(parents=True, exist_ok=True)
        current_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the config file into the current results directory
        copied_config_path = current_results_dir / config_filename
        shutil.copy2(args.config, copied_config_path)
        config_to_use = str(copied_config_path)
        
        for lr_label, lr in learning_rates:
            for architecture_name, model_script in job_variants:
                output_file = current_output_dir / f"{architecture_name}_{lr_label}.out"
                job_name = f"{args.id_name}_{k}_{lr_label}_{architecture_name}"
                
                # Prepare the dictionary of substitutions
                substitutions = {
                    "n_gpus": str(args.n_gpus),
                    "n_cpus": str(16 * args.n_gpus),
                    "hours": str(args.hours),
                    "output_file": str(output_file),
                    "job_name": job_name,
                    "config": config_to_use,
                    "lr": str(lr),
                    "results_dir": args.results_dir,
                    "id_name": args.id_name,
                    "dataset_dir": args.dataset_dir,
                    "store_penultimate": str(args.store_penultimate),
                    "k_dir": str(k),
                    "i_lr": str(lr_label),
                    "model_script": model_script
                }
                
                # Substitute the values into the job template
                job_script = job_template.safe_substitute(substitutions)
                script_filename = Path(f"run_job_{architecture_name}_{lr_label}_{k}.sh")
                
                print(f"Submitting: {job_name} (LR: {lr})")
                submit_job_script(job_script, script_filename)

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
    parser.add_argument('--skip-directory-prompt', action='store_true', 
                       help="Skip prompting for directory removal (always use existing)")
    
    # New arguments for single job submission
    parser.add_argument('--k', type=int, help="Specific k value for single job submission")
    parser.add_argument('--job-variant', type=str, choices=['lpc', 'lpc_narrow', 'no_pen'],
                       help="Job variant (lpc, lpc_narrow, or no_pen)")
    parser.add_argument('--lr-label', type=int, choices=[0, 1, 2],
                       help="Learning rate label (0=0.0001, 1=0.001, 2=5e-05)")
    parser.add_argument('--lr', type=float, help="Actual learning rate value")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Convert store_penultimate to string for the template
    args.store_penultimate = str(args.store_penultimate)
    
    # Check if this is a single job submission or batch submission
    if args.k is not None and args.job_variant is not None and args.lr_label is not None and args.lr is not None:
        # Single job submission mode (called from bash script)
        submit_single_job(args)
    else:
        # Batch submission mode (original behavior)
        if args.k is not None or args.job_variant is not None or args.lr_label is not None or args.lr is not None:
            print("Error: For single job submission, all of --k, --job-variant, --lr-label, and --lr must be specified")
            print("For batch submission, none of these should be specified")
            exit(1)
        submit_all_jobs(args)

if __name__ == '__main__':
    main()

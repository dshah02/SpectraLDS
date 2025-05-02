#!/bin/sh
#SBATCH --job-name=filter_gen_ablate # create a short name for your job
#SBATCH --output=slurm/slurm_output/fg/%x_%j.out
#SBATCH --error=slurm/slurm_output/fg/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                 # total memory per CPU core (4 GB per cpu-core is default)
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=BEGIN,END,FAIL          # send email on job start, end and on fail
#SBATCH --mail-user=ds6237@princeton.edu #--> ADD YOUR EMAIL

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.4
module load gcc-toolset/10

conda activate torch-env #--> REPLACE WITH YOUR CONDA ENV

log_info "Python version: $(python --version 2>&1)"

python -c "import torch; print(f'PyTorch version: {torch.version.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Print the GPU information
if command -v nvidia-smi &>/dev/null; then
    log_info "GPU Information: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
else
    log_info "CUDA not installed or GPUs not available."
fi

echo "Running nvidia-smi to check GPU status:"
nvidia-smi

python -m flash_stu.utils.filter_gen --sc 50 --gc 150 --shuffle_steps 1000000 --gradient_descent_steps 50000
python -m flash_stu.utils.filter_gen --sc 30 --gc 170 --shuffle_steps 1000000 --gradient_descent_steps 50000
python -m flash_stu.utils.filter_gen --sc 80 --gc 120 --shuffle_steps 1000000 --gradient_descent_steps 50000
python -m flash_stu.utils.filter_gen --sc 120 --gc 80 --shuffle_steps 1000000 --gradient_descent_steps 50000
python -m flash_stu.utils.filter_gen --sc 20 --gc 180 --shuffle_steps 1000000 --gradient_descent_steps 50000
python -m flash_stu.utils.filter_gen --sc 150 --gc 50 --shuffle_steps 1000000 --gradient_descent_steps 50000

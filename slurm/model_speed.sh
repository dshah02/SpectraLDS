#!/bin/sh
#SBATCH --job-name=all # create a short name for your job
#SBATCH --output=slurm/slurm_output/all/%x_%j.out
#SBATCH --error=slurm/slurm_output/all/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=128G                 # total memory per CPU core (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80       # Use for enforcing GPUs with 80GB memory, without it will get either 40GB or 80GB
#SBATCH --time=20:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=eladgroup
#SBATCH --partition=pli
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

python -m flash_stu.tests.inference --config_path ./configs/hybrid/naive_cache_hybrid/config.json --eval_path ./configs/hybrid/naive_cache_hybrid/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/naive_cache/config.json --eval_path ./configs/stu_only/naive_cache/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/hybrid/future_fill_hybrid/config.json --eval_path ./configs/hybrid/future_fill_hybrid/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/future_fill/config.json --eval_path ./configs/stu_only/future_fill/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/hybrid/lds_hybrid_800/config.json --eval_path ./configs/hybrid/lds_hybrid_800/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/hybrid/lds_hybrid_400/config.json --eval_path ./configs/hybrid/lds_hybrid_400/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/hybrid/lds_hybrid_200/config.json --eval_path ./configs/hybrid/lds_hybrid_200/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/hybrid/lds_hybrid_100/config.json --eval_path ./configs/hybrid/lds_hybrid_100/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/lds_800/config.json --eval_path ./configs/stu_only/lds_800/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/lds_400/config.json --eval_path ./configs/stu_only/lds_400/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/lds_200/config.json --eval_path ./configs/stu_only/lds_200/eval.yaml
python -m flash_stu.tests.inference --config_path ./configs/stu_only/lds_100/config.json --eval_path ./configs/stu_only/lds_100/eval.yaml

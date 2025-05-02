#!/bin/bash
#SBATCH --job-name=speed_test_seq # create a short name for your job
#SBATCH --output=slurm/slurm_output/april9/speed/%x_%j.out
#SBATCH --error=slurm/slurm_output/april9/speed/%x_%j.err
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G # total memory per CPU core (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80 # Use for enforcing GPUs with 80GB memory
#SBATCH --time=48:00:00 # total run time limit (HH:MM:SS) - increased for sequential execution
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --mail-type=BEGIN,END,FAIL # send email on job start, end and on fail
#SBATCH --mail-user=ds6237@princeton.edu

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.4
module load gcc-toolset/10
conda activate torch-env  # REPLACE WITH YOUR CONDA ENV

log_info "Python version: $(python --version 2>&1)"
python -c "import torch; print(f'PyTorch version: {torch.version.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

if command -v nvidia-smi &>/dev/null; then
    log_info "GPU Information: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
else
    log_info "CUDA not installed or GPUs not available."
fi

echo "Running nvidia-smi to check GPU status:"
nvidia-smi

MODELS=("stu" "stu_ff" "stu_no_approx" "distill_stu" "distill_stu_800")
SEQ_LENGTHS=(32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576)

OUTPUT_DIR="speed_test_results_$(date +%Y%m%d)"
mkdir -p $OUTPUT_DIR

log_info "Starting sequential benchmark runs. Output directory: $OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
    for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
        log_info "Running test for model: $MODEL, sequence length: $SEQ_LEN"
        
        python -m flash_stu.tests.layer_speed --model $MODEL --seq_len $SEQ_LEN --output_dir $OUTPUT_DIR
        
        # anything lingerings
        python -c "import torch; torch.cuda.empty_cache()"
        
        sleep 5
        
        log_info "Completed test for model: $MODEL, sequence length: $SEQ_LEN"
    done
done

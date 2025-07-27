#!/bin/bash

#SBATCH --job-name=vatex_eval_highperf
#SBATCH --output=logs/vatex_eval_highperf_%j.out
#SBATCH --error=logs/vatex_eval_highperf_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --cpus-per-task=48

set -e

echo "Starting MPII High-Performance Comparison Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Target: 2 GPUs, 48 CPUs, 80GB RAM for faster processing"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo ""
fi

echo "Loading modules..."
module load python/3.12
module load cuda/12.6

echo "Creating logs directory if it doesn't exist..."
mkdir -p logs

echo "Activating virtual environment..."
source /mmfs2/home/jacks.local/mali9292/scratch/Video-Comprehension-Score/venv/bin/activate

echo "Setting CUDA environment variables for multi-GPU..."
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

echo "Running high-performance vatex_eval evaluation..."
python scripts/vatex-eval.py --config config/vatex-eval.yaml

echo "High-performance vatex_eval script completed at $(date)"

if [ $? -eq 0 ]; then
    echo "SUCCESS: High-performance vatex_eval evaluation completed successfully"
else
    echo "ERROR: High-performance vatex_eval evaluation failed"
    exit 1
fi
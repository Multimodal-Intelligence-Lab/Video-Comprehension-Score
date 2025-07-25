#!/bin/bash

#SBATCH --job-name=mpii_vlm_eval_highperf
#SBATCH --output=logs/vlm_eval_highperf_%j.out
#SBATCH --error=logs/vlm_eval_highperf_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=all-gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
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
source /mmfs2/home/jacks.local/mali9292/scratch/dev-vcs-vatex/vcs/venv/bin/activate

echo "Setting CUDA environment variables for multi-GPU..."
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

echo "Running high-performance vlm_eval evaluation..."
python scripts/clipcc_eval_vlms.py --config config/clipcc_eval_vlms.yaml

echo "High-performance vlm_eval script completed at $(date)"

if [ $? -eq 0 ]; then
    echo "SUCCESS: High-performance vlm_eval evaluation completed successfully"
else
    echo "ERROR: High-performance vlm_eval evaluation failed"
    exit 1
fi
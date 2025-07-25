#!/bin/bash

#SBATCH --job-name=mpii_comparison_highperf
#SBATCH --output=logs/comparison_highperf_%j.out
#SBATCH --error=logs/comparison_highperf_%j.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32

set -e

echo "Starting MPII High-Performance Comparison Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Target: 2 GPUs, 32 CPUs, 256GB RAM for faster processing"

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo ""
fi

echo "Loading modules..."
module load python/3.8
module load cuda/11.8

echo "Creating logs directory if it doesn't exist..."
mkdir -p logs

echo "Activating virtual environment..."
source /mmfs1/scratch/jacks.local/mali9292/vcs/venv/bin/activate

echo "Setting CUDA environment variables for multi-GPU..."
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

echo "Running high-performance comparison evaluation..."
python mpii_eval_comparison.py --config config/comparison.yaml

echo "High-performance comparison script completed at $(date)"

if [ $? -eq 0 ]; then
    echo "SUCCESS: High-performance comparison evaluation completed successfully"
else
    echo "ERROR: High-performance comparison evaluation failed"
    exit 1
fi
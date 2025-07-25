#!/bin/bash

#SBATCH --job-name=mpii_comparison_test
#SBATCH --output=logs/comparison_test_%j.out
#SBATCH --error=logs/comparison_test_%j.err
#SBATCH --time=5:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8

set -e

echo "Starting MPII Comparison Test Script at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"

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

echo "Setting CUDA environment variables..."
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

echo "Running comparison evaluation test..."
python mpii_eval_comparison.py --config config/comparison.yaml

echo "Test comparison script completed at $(date)"

if [ $? -eq 0 ]; then
    echo "SUCCESS: Test comparison evaluation completed successfully"
else
    echo "ERROR: Test comparison evaluation failed"
    exit 1
fi
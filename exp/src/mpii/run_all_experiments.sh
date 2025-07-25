#!/bin/bash

# MPII Evaluation Pipeline - Run All Experiments
# This script runs all MPII experiments (comparison, addition, deletion, chronology, authors) sequentially
# 
# Usage: ./run_all_experiments.sh [--verbose]
# 
# The script will:
# 1. Run each experiment with its corresponding config file
# 2. Log all outputs with timestamps
# 3. Provide a summary at the end
# 4. Stop on any experiment failure (unless --continue-on-error is used)

set -e  # Exit on any error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
VERBOSE=false
CONTINUE_ON_ERROR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--verbose] [--continue-on-error]"
            echo ""
            echo "Options:"
            echo "  --verbose           Enable verbose output for all experiments"
            echo "  --continue-on-error Continue running remaining experiments if one fails"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Define experiments in order
EXPERIMENTS=(
    "comparison:mpii_eval_comparison.py:config/comparison.yaml"
    "addition:mpii_eval_addition.py:config/addition.yaml"
    "deletion:mpii_eval_deletion.py:config/deletion.yaml"
    "chronology:mpii_eval_chronology.py:config/chronology.yaml"
    "authors:mpii_eval_authors.py:config/authors.yaml"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Initialize summary
TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
SUCCESSFUL_EXPERIMENTS=0
FAILED_EXPERIMENTS=()
START_TIME=$(date +%s)

echo "========================================="
echo "MPII Evaluation Pipeline - All Experiments"
echo "========================================="
echo "Starting time: $(date)"
echo "Total experiments: $TOTAL_EXPERIMENTS"
echo "Working directory: $SCRIPT_DIR"
echo "Verbose mode: $VERBOSE"
echo "Continue on error: $CONTINUE_ON_ERROR"
echo "========================================="
echo ""

# Function to run a single experiment
run_experiment() {
    local exp_name="$1"
    local script_name="$2"
    local config_file="$3"
    local exp_num="$4"
    
    echo "[$exp_num/$TOTAL_EXPERIMENTS] Starting $exp_name experiment..."
    echo "Script: $script_name"
    echo "Config: $config_file"
    echo "Time: $(date)"
    
    # Check if files exist
    if [[ ! -f "$script_name" ]]; then
        echo "ERROR: Script not found: $script_name"
        return 1
    fi
    
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file not found: $config_file"
        return 1
    fi
    
    # Build command
    local cmd="python $script_name --config $config_file"
    if [[ "$VERBOSE" == "true" ]]; then
        cmd="$cmd --verbose"
    fi
    
    # Create experiment-specific log file
    local log_file="logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Command: $cmd"
    echo "Log file: $log_file"
    echo ""
    
    # Run the experiment
    local exp_start_time=$(date +%s)
    
    if $cmd 2>&1 | tee "$log_file"; then
        local exp_end_time=$(date +%s)
        local exp_duration=$((exp_end_time - exp_start_time))
        echo ""
        echo "✅ $exp_name experiment completed successfully!"
        echo "Duration: ${exp_duration}s ($(date -u -d @${exp_duration} +%H:%M:%S))"
        echo "Log saved to: $log_file"
        return 0
    else
        local exp_end_time=$(date +%s)
        local exp_duration=$((exp_end_time - exp_start_time))
        echo ""
        echo "❌ $exp_name experiment FAILED!"
        echo "Duration: ${exp_duration}s ($(date -u -d @${exp_duration} +%H:%M:%S))"
        echo "Check log file for details: $log_file"
        return 1
    fi
}

# Run all experiments
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r exp_name script_name config_file <<< "${EXPERIMENTS[$i]}"
    exp_num=$((i + 1))
    
    if run_experiment "$exp_name" "$script_name" "$config_file" "$exp_num"; then
        SUCCESSFUL_EXPERIMENTS=$((SUCCESSFUL_EXPERIMENTS + 1))
    else
        FAILED_EXPERIMENTS+=("$exp_name")
        
        if [[ "$CONTINUE_ON_ERROR" == "false" ]]; then
            echo ""
            echo "❌ Stopping due to failure in $exp_name experiment."
            echo "Use --continue-on-error to continue running remaining experiments."
            break
        fi
    fi
    
    # Add separator between experiments (except for the last one)
    if [[ $exp_num -lt $TOTAL_EXPERIMENTS ]]; then
        echo ""
        echo "========================================="
        echo ""
    fi
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================="
echo "MPII EVALUATION PIPELINE SUMMARY"
echo "========================================="
echo "Completion time: $(date)"
echo "Total duration: ${TOTAL_DURATION}s ($(date -u -d @${TOTAL_DURATION} +%H:%M:%S))"
echo ""
echo "RESULTS:"
echo "✅ Successful experiments: $SUCCESSFUL_EXPERIMENTS/$TOTAL_EXPERIMENTS"

if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    echo "❌ Failed experiments: ${#FAILED_EXPERIMENTS[@]}/$TOTAL_EXPERIMENTS"
    echo "Failed experiments:"
    for failed_exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $failed_exp"
    done
else
    echo "🎉 All experiments completed successfully!"
fi

echo ""
echo "Log files saved in: $SCRIPT_DIR/logs/"
echo "Results saved in: $SCRIPT_DIR/../../results/mpii/"

# Exit with appropriate code
if [[ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]]; then
    exit 1
else
    exit 0
fi
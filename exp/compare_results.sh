#!/bin/bash

# Script to compare experimental results before and after refactoring
# Usage: ./compare_results.sh [baseline_suffix]
# 
# Examples:
#   ./compare_results.sh                    # Auto-detects latest baseline
#   ./compare_results.sh _20250712          # Compare with specific baseline
#   ./compare_results.sh _backup            # Compare with backup

# Get script directory to work from any location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Determine baseline directory
if [ -n "$1" ]; then
    # Use provided suffix
    BASELINE_SUFFIX="$1"
    BASELINE_DIR="$RESULTS_DIR/mpii_baseline$BASELINE_SUFFIX"
else
    # Auto-detect latest baseline directory
    BASELINE_DIR=$(find "$RESULTS_DIR" -maxdepth 1 -type d -name "mpii_baseline*" | sort -r | head -1)
    if [ -z "$BASELINE_DIR" ]; then
        echo "ERROR: No baseline directory found!"
        echo "Expected: $RESULTS_DIR/mpii_baseline*"
        echo ""
        echo "Available directories:"
        ls -la "$RESULTS_DIR" | grep "mpii"
        exit 1
    fi
    BASELINE_SUFFIX="${BASELINE_DIR#$RESULTS_DIR/mpii_baseline}"
fi

CURRENT_DIR="$RESULTS_DIR/mpii"

echo "========================================="
echo "Comparing experimental results:"
echo "Baseline: $(basename "$BASELINE_DIR")"
echo "Current:  $(basename "$CURRENT_DIR")"
echo "Working from: $SCRIPT_DIR"
echo "========================================="
echo

# Check if baseline directory exists
if [ ! -d "$BASELINE_DIR" ]; then
    echo "ERROR: Baseline directory not found!"
    echo "Expected: $BASELINE_DIR"
    echo ""
    echo "Available baseline directories:"
    find "$RESULTS_DIR" -maxdepth 1 -type d -name "mpii_baseline*" | sort
    exit 1
fi

# Check if current directory exists
if [ ! -d "$CURRENT_DIR" ]; then
    echo "ERROR: Current results directory not found!"
    echo "Expected: $CURRENT_DIR"
    exit 1
fi

# Initialize counters
identical=0
different=0
missing=0
total=0

echo "Checking CSV files..."
echo

# Find and compare all CSV files, excluding .ipynb_checkpoints
while IFS= read -r -d '' baseline_file; do
    # Skip checkpoint files
    if [[ "$baseline_file" == *".ipynb_checkpoints"* ]]; then
        continue
    fi
    
    # Convert baseline path to current path
    current_file="${baseline_file/$BASELINE_DIR/$CURRENT_DIR}"
    relative_path="${baseline_file#$BASELINE_DIR/}"
    
    total=$((total + 1))
    
    if [ -f "$current_file" ]; then
        if diff -q "$baseline_file" "$current_file" > /dev/null 2>&1; then
            echo "✓ $relative_path - identical"
            echo "  Baseline: $(basename "$BASELINE_DIR")/$relative_path"
            echo "  Current:  $(basename "$CURRENT_DIR")/$relative_path"
            identical=$((identical + 1))
        else
            echo "✗ $relative_path - DIFFERENT"
            echo "  Baseline: $(basename "$BASELINE_DIR")/$relative_path"
            echo "  Current:  $(basename "$CURRENT_DIR")/$relative_path"
            echo "  To see differences: diff '$baseline_file' '$current_file'"
            different=$((different + 1))
        fi
    else
        echo "? $relative_path - missing in current results"
        echo "  Expected: $(basename "$CURRENT_DIR")/$relative_path"
        missing=$((missing + 1))
    fi
done < <(find "$BASELINE_DIR" -name "*.csv" -print0)

echo
echo "========================================="
echo "Comparison complete!"
echo "========================================="
echo "SUMMARY:"
echo "Total files compared: $total"
echo "✓ Identical: $identical"
echo "✗ Different: $different"
echo "? Missing: $missing"
echo

if [ $different -eq 0 ] && [ $missing -eq 0 ]; then
    echo "🎉 SUCCESS: All files are identical!"
    echo "✅ Refactoring completed successfully with no regressions."
elif [ $different -eq 0 ]; then
    echo "⚠️  WARNING: Some files are missing but existing files are identical."
else
    echo "❌ ATTENTION: Some files have differences that need investigation."
fi

echo
echo "Legend:"
echo "✓ = Files are identical"
echo "✗ = Files have differences" 
echo "? = File missing in current results"
echo
echo "To investigate specific differences, use:"
echo "  diff 'baseline_file' 'current_file'"
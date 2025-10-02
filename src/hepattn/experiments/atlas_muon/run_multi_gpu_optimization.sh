#!/bin/bash
"""
Example script for running hyperparameter optimization across multiple GPUs.

This script starts optimization processes on GPUs 0, 1, and 2 simultaneously,
all contributing to the same Optuna study.
"""

# Configuration
STUDY_NAME="atlas_muon_optimization_$(date +%Y%m%d_%H%M%S)"
N_TRIALS=150
DB_PATH="./optuna_study_$(date +%Y%m%d_%H%M%S).db"
N_WARMUP=50

echo "Starting multi-GPU optimization..."
echo "Study name: $STUDY_NAME"
echo "Database: $DB_PATH"
echo "Trials per GPU: $N_TRIALS"
echo "Warmup trials: $N_WARMUP"
echo

# Start optimization on multiple GPUs in background
echo "Starting optimization on GPU 0..."
python hepattn/experiments/atlas_muon/optuna_tune.py \
    --gpu 0 \
    --n-trials $N_TRIALS \
    --study-name $STUDY_NAME \
    --db-path $DB_PATH \
    --n-warmup $N_WARMUP \
    --pruner &
GPU0_PID=$!

echo "Starting optimization on GPU 1..."
python hepattn/experiments/atlas_muon/optuna_tune.py \
    --gpu 1 \
    --n-trials $N_TRIALS \
    --study-name $STUDY_NAME \
    --db-path $DB_PATH \
    --n-warmup $N_WARMUP \
    --pruner &
GPU1_PID=$!

echo "Starting optimization on GPU 2..."
python hepattn/experiments/atlas_muon/optuna_tune.py \
    --gpu 2 \
    --n-trials $N_TRIALS \
    --study-name $STUDY_NAME \
    --db-path $DB_PATH \
    --n-warmup $N_WARMUP \
    --pruner &
GPU2_PID=$!

echo "All optimization processes started."
echo "GPU 0 PID: $GPU0_PID"
echo "GPU 1 PID: $GPU1_PID" 
echo "GPU 2 PID: $GPU2_PID"
echo

# Function to check if process is still running
check_process() {
    if ps -p $1 > /dev/null 2>&1; then
        return 0  # Process is running
    else
        return 1  # Process is not running
    fi
}

# Monitor progress
echo "Monitoring optimization progress..."
echo "Use Ctrl+C to stop all processes gracefully"

# Trap Ctrl+C to clean up background processes
trap 'echo "Stopping all optimization processes..."; kill $GPU0_PID $GPU1_PID $GPU2_PID 2>/dev/null; exit' INT

# Monitor until all processes complete
while check_process $GPU0_PID || check_process $GPU1_PID || check_process $GPU2_PID; do
    # Check database for progress every 30 seconds
    if command -v sqlite3 >/dev/null 2>&1 && [ -f "$DB_PATH" ]; then
        COMPLETED=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE';" 2>/dev/null || echo "0")
        RUNNING=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state = 'RUNNING';" 2>/dev/null || echo "0")
        FAILED=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state = 'FAIL';" 2>/dev/null || echo "0")
        
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Completed: $COMPLETED, Running: $RUNNING, Failed: $FAILED"
    fi
    
    sleep 30
done

echo
echo "All optimization processes completed!"
echo "Study name: $STUDY_NAME"
echo "Database: $DB_PATH"

# Show final results if sqlite3 is available
if command -v sqlite3 >/dev/null 2>&1 && [ -f "$DB_PATH" ]; then
    echo
    echo "Final Results:"
    COMPLETED=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state = 'COMPLETE';" 2>/dev/null || echo "0")
    FAILED=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM trials WHERE state = 'FAIL';" 2>/dev/null || echo "0")
    echo "Completed trials: $COMPLETED"
    echo "Failed trials: $FAILED"
    
    # Show best trial if available
    BEST_VALUE=$(sqlite3 "$DB_PATH" "SELECT MIN(value) FROM trials WHERE state = 'COMPLETE';" 2>/dev/null || echo "N/A")
    if [ "$BEST_VALUE" != "N/A" ] && [ "$BEST_VALUE" != "" ]; then
        echo "Best objective value: $BEST_VALUE"
    fi
    
    echo
    echo "To analyze results, run:"
    echo "python hepattn/experiments/atlas_muon/analyze_results.py --db-path $DB_PATH --study-name $STUDY_NAME"
fi
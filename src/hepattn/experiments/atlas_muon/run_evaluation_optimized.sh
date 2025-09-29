#!/bin/bash
#
# OPTIMIZED ATLAS Muon Hit Filter Evaluation Runner
# This script runs the evaluation with proper background execution and logging
#

# Default parameters - modify as needed
EVAL_PATH="/shared/tracking/data/1_5Mio_test_training/epoch=049-val_acc=0.99711_ml_test_data_150K_processed_eval.h5"
DATA_DIR="/shared/tracking/data/ml_test_data_150K_processed"
CONFIG_PATH="./hepattn/experiments/atlas_muon/configs/NGT/atlas_muon_event_NGT_plotting.yaml"
OUTPUT_DIR="./evaluation_results_optimized"
MAX_EVENTS=-1  # -1 for all events

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_path)
            EVAL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_events)
            MAX_EVENTS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --eval_path PATH      Path to evaluation HDF5 file"
            echo "  --data_dir PATH       Path to processed data directory"
            echo "  --config_path PATH    Path to config YAML file"
            echo "  --output_dir PATH     Output directory"
            echo "  --max_events N        Maximum events to process (-1 for all)"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate unique log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/evaluation_${TIMESTAMP}.log"

echo "=========================================="
echo "ATLAS Muon Hit Filter Evaluation (OPTIMIZED)"
echo "=========================================="
echo "Evaluation file: $EVAL_PATH"
echo "Data directory: $DATA_DIR"
echo "Config file: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Max events: $MAX_EVENTS"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Check if files exist
if [[ ! -f "$EVAL_PATH" ]]; then
    echo "ERROR: Evaluation file not found: $EVAL_PATH"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Set up Python environment if needed
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found in PATH"
    exit 1
fi

# Build the command
CMD="$PYTHON_CMD evaluate_hit_filter_dataloader_optimized.py \
    --eval_path '$EVAL_PATH' \
    --data_dir '$DATA_DIR' \
    --config_path '$CONFIG_PATH' \
    --output_dir '$OUTPUT_DIR' \
    --max_events $MAX_EVENTS"

echo "Command to execute:"
echo "$CMD"
echo ""

# Ask user if they want to run in background
read -p "Run in background with nohup? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting evaluation in background..."
    echo "Log file: $LOG_FILE"
    echo "To monitor progress: tail -f $LOG_FILE"
    echo "To check if running: ps aux | grep evaluate_hit_filter"
    echo ""
    
    # Run with nohup and redirect output
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo "Background process started with PID: $PID"
    echo "To kill the process if needed: kill $PID"
    
    # Wait a moment and check if process is still running
    sleep 2
    if kill -0 $PID 2>/dev/null; then
        echo "Process is running successfully."
        echo "Check the log file for progress: tail -f $LOG_FILE"
    else
        echo "WARNING: Process may have failed to start. Check the log file: $LOG_FILE"
    fi
else
    echo "Running in foreground..."
    echo ""
    # Run in foreground with tee to show output and save to log
    bash -c "$CMD" 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "Evaluation script completed."
echo "Results will be in: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"

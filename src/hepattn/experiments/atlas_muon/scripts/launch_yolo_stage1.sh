#!/bin/bash
# Launch all YOLO Stage 1 classification training runs on GPUs 0-3
#
# Usage:
#   cd hepattn_muon/src
#   ./hepattn/experiments/atlas_muon/scripts/launch_yolo_stage1.sh
#
# This launches 4 parallel training runs with different bin resolutions:
#   GPU 0: Coarse   (pt=50,  eta/phi=100)
#   GPU 1: Medium   (pt=100, eta/phi=2000)
#   GPU 2: High     (pt=200, eta/phi=6000)
#   GPU 3: Ultra    (pt=400, eta/phi=12000)

set -e

CONFIG_DIR="hepattn/experiments/atlas_muon/configs/YOLO"
LOG_DIR="logs/YOLO/stage1_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

echo "Starting YOLO Stage 1 Classification Training"
echo "=============================================="
echo "Log directory: $LOG_DIR"
echo ""

# GPU 0: Coarse resolution
echo "Launching GPU 0 (Coarse: pt=50, eta/phi=100)..."
nohup pixi run python -m hepattn.experiments.atlas_muon.run_yolo_classification \
    --config "$CONFIG_DIR/stage1_cls_gpu0.yaml" \
    > "$LOG_DIR/gpu0_coarse.log" 2>&1 &
PID0=$!
echo "  PID: $PID0"

# GPU 1: Medium resolution
echo "Launching GPU 1 (Medium: pt=100, eta/phi=2000)..."
nohup pixi run python -m hepattn.experiments.atlas_muon.run_yolo_classification \
    --config "$CONFIG_DIR/stage1_cls_gpu1.yaml" \
    > "$LOG_DIR/gpu1_medium.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

# GPU 2: High resolution
echo "Launching GPU 2 (High: pt=200, eta/phi=6000)..."
nohup pixi run python -m hepattn.experiments.atlas_muon.run_yolo_classification \
    --config "$CONFIG_DIR/stage1_cls_gpu2.yaml" \
    > "$LOG_DIR/gpu2_high.log" 2>&1 &
PID2=$!
echo "  PID: $PID2"

# GPU 3: Ultra resolution
echo "Launching GPU 3 (Ultra: pt=400, eta/phi=12000)..."
nohup pixi run python -m hepattn.experiments.atlas_muon.run_yolo_classification \
    --config "$CONFIG_DIR/stage1_cls_gpu3.yaml" \
    > "$LOG_DIR/gpu3_ultra.log" 2>&1 &
PID3=$!
echo "  PID: $PID3"

echo ""
echo "All training runs launched!"
echo "PIDs: $PID0, $PID1, $PID2, $PID3"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/gpu0_coarse.log"
echo "  tail -f $LOG_DIR/gpu1_medium.log"
echo "  tail -f $LOG_DIR/gpu2_high.log"
echo "  tail -f $LOG_DIR/gpu3_ultra.log"
echo ""
echo "Or watch all with:"
echo "  watch -n 5 'nvidia-smi'"

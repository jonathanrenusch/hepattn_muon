#!/usr/bin/env bash
# ============================================================================
# Generate all 8 dataset variants (4 pairs: pretrain on p0, finetune on p200)
# ============================================================================
#
# Pretraining uses the p0 (no pileup) sample — ~6x more hard-scatter tracks
# per event, no pileup contamination.  Fine-tuning uses p200 (200 pileup) for
# realistic detector occupancy.
#
# The four base variants are: loose, core, core_kf_matched, core_kf_hits.
# For pretrain the hard_scatter cut is forced true (signal only).
# For finetune it is forced false (all primaries incl. pileup).
#
# Output directories:
#   /eos/project/e/end-to-end-colliderml/data/NeurIPS_retraining/
#     p0_{variant}_pretrain      (from p0 sample)
#     p200_{variant}_finetune    (from p200 sample)
#
# Usage:
#   # All 8 datasets sequentially on one machine:
#   ./run_all_p200_datasets.sh
#
#   # Only specific variants (space-separated):
#   ./run_all_p200_datasets.sh loose core
#
#   # Only pretrain or only finetune stage:
#   STAGES="pretrain"           ./run_all_p200_datasets.sh
#   STAGES="finetune"           ./run_all_p200_datasets.sh loose
#
#   # Custom workers:
#   WORKERS=8 ./run_all_p200_datasets.sh
#
#   # Quick test (2 shards):
#   NUM_SHARDS=2 ./run_all_p200_datasets.sh
#
# Run from the hepattn_muon directory with pixi, or set PIXI_CMD:
#   cd /shared/tracking/hepattn_muon
#   ./src/hepattn/experiments/colliderml_regr/scripts/run_all_p200_datasets.sh
#
# For multi-machine parallelism, run different variants on different machines:
#   Machine 1: ./run_all_p200_datasets.sh loose
#   Machine 2: ./run_all_p200_datasets.sh core
#   Machine 3: ./run_all_p200_datasets.sh core_kf_matched
#   Machine 4: ./run_all_p200_datasets.sh core_kf_hits
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

PREPROCESS_SCRIPT="${SCRIPT_DIR}/preprocess_colliderml_compact.py"
SEL_FILE="${SCRIPT_DIR}/../utils/selection_p200_datasets.yaml"

# ── Source data directories ──
P0_DATA_DIR="${P0_DATA_DIR:-/eos/project/e/end-to-end-muon-tracking/tracking/colliderml/p0/CERN__ColliderML-Release-1}"
P200_DATA_DIR="${P200_DATA_DIR:-/eos/project/n/ngt2-4/data/ColliderML-Release-1.old/data}"

# ── Output base ──
OUT_BASE="${OUT_BASE:-/eos/project/e/end-to-end-colliderml/data/NeurIPS_retraining}"

WORKERS="${WORKERS:-16}"
NUM_SHARDS="${NUM_SHARDS:--1}"
PIXI_CMD="${PIXI_CMD:-pixi run}"
STAGES="${STAGES:-pretrain finetune}"

# Variants to process (default: all four)
if [[ $# -gt 0 ]]; then
    VARIANTS=("$@")
else
    VARIANTS=(loose core core_kf_matched core_kf_hits)
fi

echo "========================================================================"
echo "Dataset Generation (pretrain=p0, finetune=p200)"
echo "========================================================================"
echo "  P0 data dir:    ${P0_DATA_DIR}"
echo "  P200 data dir:  ${P200_DATA_DIR}"
echo "  Output base:    ${OUT_BASE}"
echo "  Workers:        ${WORKERS}"
echo "  Num shards:     ${NUM_SHARDS}"
echo "  Variants:       ${VARIANTS[*]}"
echo "  Stages:         ${STAGES}"
echo "  Selection:      ${SEL_FILE}"
echo "========================================================================"
echo ""

cd "${REPO_ROOT}"

for variant in "${VARIANTS[@]}"; do
    for stage in ${STAGES}; do
        if [[ "${stage}" == "pretrain" ]]; then
            hs_val="true"
            data_dir="${P0_DATA_DIR}"
            particles_subdir="ttbar_pu0_particles_recorded_only"
            hits_subdir="ttbar_pu0_tracker_hits"
            tracks_subdir="ttbar_pu0_tracks"
            out_dir="${OUT_BASE}/p0_${variant}_pretrain"
        elif [[ "${stage}" == "finetune" ]]; then
            hs_val="false"
            data_dir="${P200_DATA_DIR}"
            particles_subdir="ttbar_pu200_particles_recorded_only"
            hits_subdir="ttbar_pu200_tracker_hits"
            tracks_subdir="ttbar_pu200_tracks"
            out_dir="${OUT_BASE}/p200_${variant}_finetune"
        else
            echo "Unknown stage: ${stage}"; exit 1
        fi

        echo "================================================================"
        echo "  ${variant} / ${stage} -> ${out_dir}"
        echo "================================================================"

        ${PIXI_CMD} python "${PREPROCESS_SCRIPT}" \
            --data-dir "${data_dir}" \
            --particles-subdir "${particles_subdir}" \
            --hits-subdir "${hits_subdir}" \
            --tracks-subdir "${tracks_subdir}" \
            --output-dir "${out_dir}" \
            --selection-file "${SEL_FILE}" \
            --selection-variant "${variant}" \
            --selection "{\"hard_scatter\": ${hs_val}}" \
            --num-shards "${NUM_SHARDS}" \
            --num-workers "${WORKERS}"

        echo ""
    done
done

echo "========================================================================"
echo "All dataset generation complete."
echo "========================================================================"
echo ""
echo "Visualization commands:"
for variant in "${VARIANTS[@]}"; do
    for stage in ${STAGES}; do
        if [[ "${stage}" == "pretrain" ]]; then
            ds="p0_${variant}_pretrain"
        else
            ds="p200_${variant}_finetune"
        fi
        echo "  ${PIXI_CMD} python ${PREPROCESS_SCRIPT%/*}/visualize_dataset.py \\"
        echo "    --preprocessed-dir ${OUT_BASE}/${ds} --output-dir /tmp/viz_${ds}"
    done
done

#!/bin/bash
# Run the paper-plot pipeline for N runs in parallel across GPUs.
# Edit RUNS to add rows; one row per (gpu, run_id, nicename, optional_d0_run_id, axes...).
set -u

cd /shared/tracking/hepattn_muon || exit 1

run_one() {
  local gpu=$1; shift
  local run_id=$1; shift
  local nicename=$1; shift
  local d0_run_id=$1; shift
  local axes=("$@")

  local logdir=/shared/tracking/logs_Neurips/paper_plots/${nicename}
  mkdir -p "$logdir"
  local log=${logdir}/pipeline.log

  local d0_arg=""
  [ -n "$d0_run_id" ] && d0_arg="--d0-run-id $d0_run_id"
  local axes_arg=""
  if [ ${#axes[@]} -gt 0 ]; then
    axes_arg="--ablation-axes ${axes[*]}"
  fi

  echo "[$nicename] gpu=$gpu start $(date -Is)" | tee "$log"
  CUDA_VISIBLE_DEVICES=$gpu pixi run python -m \
    hepattn.experiments.colliderml_regr.paper_plots.cli \
    --run-id "$run_id" \
    --nicename "$nicename" \
    --gpu "$gpu" \
    --skip-inference \
    --skip-aggregate \
    $d0_arg $axes_arg >> "$log" 2>&1
  local rc=$?
  echo "[$nicename] rc=$rc $(date -Is)" >> "$log"
  return $rc
}

# ---- edit below per launch ----
RUNS=(
  "0 7972d00dcde44bb199bfdf4c870587a5 ssmcls_q7_p0pretrain_zeroshot_7972d00d_ep49 '' pooling"
  "1 5c45569953984eed9bf010a1e8700459 ssmcls_q7_kfmatched_finetune_5c455699_ep46 '' finetune"
  "2 ee806c6fe2ec4808baafbb35bf4f24bd txf_q7_p0pretrain_zeroshot_ee806c6f_ep46    '' transformer"
)

PIDS=()
for row in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  eval "run_one $row" &
  PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || RC=$?
done

# Aggregator at the end (single pass over the whole tree)
pixi run python -m hepattn.experiments.colliderml_regr.paper_plots.aggregate
echo "all done; rc=$RC"
exit $RC

#!/bin/bash
# Paper-plot pipeline batch — 2026-04-29 (4× H100, full machine).
#
# Naming convention (NEW, 2026-04-29):
#   <full-comet-experiment-name>__<runhash8>__ep<NN>
# The full Comet experiment name (read from config.yaml -> trainer.logger.name)
# is preserved verbatim so runs stay distinguishable in /shared/tracking/logs_Neurips/paper_plots/.
# The 8-char run-hash prefix disambiguates if two runs share a comet name.
#
# Inference is NOT skipped: --skip-inference omitted so cli.py spawns
# train.py test on the assigned GPU when the h5 is missing.
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
    --skip-aggregate \
    $d0_arg $axes_arg >> "$log" 2>&1
  local rc=$?
  echo "[$nicename] rc=$rc $(date -Is)" >> "$log"
  return $rc
}

# ---- 2026-04-29 batch ----
RUNS=(
  "0 f7d7198a5df54bc4a0ae0501ad060152 TRK-v2-SSMCLS-10L-sepD0DFL-gradnorm001-warm-FP32__f7d7198a__ep29 '' d0_head loss_design"
  "1 5c45569953984eed9bf010a1e8700459 TRK-v2-SSMCLS-Q7-A-AdamW-WSD-FP32-finetune__5c455699__ep49 '' finetune"
  "2 ee22ead18ad9410c9c032dcc82501be7 TRK-v2-SSMCLS-Q7-B-Lion-WSD-FP32-finetune__ee22ead1__ep49 '' finetune"
  "3 82df4d9662fb4ca19b912b37e9fd5c22 TRK-v2-SSMCLS-scaling-run3-15L-dstate64-allQuant__82df4d96__ep49 '' scaling"
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

pixi run python -m hepattn.experiments.colliderml_regr.paper_plots.aggregate
echo "all done; rc=$RC"
exit $RC

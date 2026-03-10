#!/bin/bash
set -e
export TRITON_CACHE_DIR=/tmp/triton_cache
PYTHON=/shared/tracking/hepattn_muon/.pixi/envs/default/bin/python
TRAIN=/shared/tracking/hepattn_muon/src/hepattn/experiments/colliderml_regr/train.py
CFG_DIR=/shared/tracking/hepattn_muon/src/hepattn/experiments/colliderml_regr/config

COMMON="--trainer.max_epochs=2 --trainer.logger=false --trainer.enable_checkpointing=false --trainer.devices=[0] --trainer.callbacks=null --data.preprocessed_dir=/scratch/colliderml/p0_preprocessed_test --data.num_shards=2 --data.num_workers=2 --data.batch_size=2048 --data.train_frac=0.5 --data.val_frac=0.5 --model.optimizer=AdamW --model.lrs_config.max=5.0e-04 --model.lrs_config.weight_decay=1.0e-04"

for study in study1_baseline_smooth_l1 study2_spline_l1 study3_spline_quantile study4_direct_quantile; do
    echo "=== ${study} ==="
    DIR=/tmp/test_${study}
    rm -rf "$DIR"
    mkdir -p "$DIR"
    $PYTHON $TRAIN fit --config ${CFG_DIR}/${study}.yaml --trainer.default_root_dir="$DIR" $COMMON 2>&1 | tail -3
    echo
done
echo "ALL DONE"

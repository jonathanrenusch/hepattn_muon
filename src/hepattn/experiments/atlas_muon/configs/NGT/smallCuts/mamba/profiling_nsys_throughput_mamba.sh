#!/bin/bash
nsys profile \
  --trace=cuda,osrt \
  --sample=cpu \
  --force-overwrite true \
  -o profile_mamba_bi2_B128 \
  pixi run python -m hepattn.experiments.atlas_muon.run_filtering test -c /shared/ML/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/smallCuts/mamba/atlas_muon_filtering_mamba_bidirectional_2infer.yaml
#!/bin/bash
nsys profile \
  --trace=cuda,osrt \
  --sample=cpu \
  -o profile_transf_B128 \
  pixi run python -m hepattn.experiments.atlas_muon.run_filtering test -c /shared/ML/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/NGT/smallCuts/transformer/atlas_muon_filtering_inf.yaml
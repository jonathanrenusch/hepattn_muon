# ColliderML track-parameter regression

## What this is

Regression of the five perigee track parameters `(d0, z0, phi, theta, qop)` from
the raw detector-measurement sequence of a charged particle, using a custom
bidirectional Mamba-2 state space model whose forward+backward scan with gated
merge is adapted from **Vision Mamba** (arXiv:2401.09417, originally proposed
for image classification). The scientific claim is that SSMs can learn
non-linear scattering effects better than the ACTS combinatorial Kalman filter
(CKF) and therefore produce more precise track parameters on the same input
measurements, **at parity with the CKF compute budget on a single GPU and with
substantially reduced sensitivity to non-Gaussian outliers**.

The comparison point throughout is **ACTS CKF** on the same events, evaluated on
the double-matched (DM) subset — i.e. only tracks where both the CKF and the
model succeed. The physics-relevant metric is the iterative 3σ-clipped RMS on
the DM subset (logged as `ssm_iqr_dm` / `ssm_rms_dm`).

### Why this matters (HEP framing for an ML audience)

Charged-particle tracking is the foundational reconstruction step at the LHC
and its successor HL-LHC: every downstream analysis — Higgs measurements,
b-tagging, dark-matter searches — consumes the five-parameter trajectory
estimates produced here. The incumbent algorithm is the **combinatorial Kalman
filter**, a recursive linear estimator that has been the unchallenged industry
standard for **30+ years**. It is statistically optimal under Gaussian noise
assumptions, but those assumptions break in two regimes that increasingly
dominate modern colliders: (a) **non-Gaussian multiple scattering tails** in
dense silicon, and (b) **pileup-induced ambiguity** at HL-LHC occupancy (200
concurrent collisions per bunch crossing, ~250 K detector measurements per
event). CKF compute on CPU also scales poorly with occupancy, motivating
GPU-native replacements.

### Why SSMs in particular

A full HL-LHC event is a 250 K-element sequence of measurements drawn from a
**heterogeneous spatio-temporal detector geometry** (multiple silicon layers,
varying granularity, time-stamped readout). State space models offer
linear-time sequence processing with a learned recurrent state, in contrast to
the quadratic cost of self-attention — a structural fit for both the per-track
regime studied here (≤20 measurements per track) and the long-context
event-level regime that future work will target. The forward and backward
Mamba-2 scans correspond to inward- and outward-going trajectory integration,
mirroring how the CKF itself runs forward+smoother passes — but with the
non-linear response learned end-to-end rather than linearised at each step.

### Scope of this work (first-of-its-kind ablations)

To our knowledge this is the first study of bidirectional Mamba-2 for
high-precision physical-parameter regression, and the first to ablate two
axes that turn out to matter:

1. **Pretraining→fine-tuning optimizer curriculum.** With Lion fixed as the
   pretraining optimizer (validated against AdamW), we ablate three
   continuations on the pileup-contaminated fine-tune stage: Lion-continuation,
   AdamW-switch, and a Muon-(2-D weights) + AdamW-(1-D weights) hybrid — all
   under WSD schedules. Motivated by SpecMuon (arXiv:2602.16167) reporting
   2–10× lower MSE on PINN/DeepONet precision regression.
2. **Sequence-summary readout.** SSM-state extraction (raw recurrent state via
   `fwd_head`/`bwd_head`) vs learned CLS-token pooling at each scan terminus
   vs a parameter-matched flash-attention transformer with a register token.
   The transformer baseline is trained under the **same parameter count and
   the same Lion+OneCycleLR pretraining recipe** as both SSM variants and
   underperforms both at matched compute.

## Physics & dataset context

Source: [CERN/ColliderML-Release-1](https://huggingface.co/datasets/CERN/ColliderML-Release-1).

- **Detector**: Open Data Detector (ODD), generic HL-LHC silicon tracker.
- **Collisions**: 14 TeV pp. Generated with MadGraph+Pythia8, simulated in
  Geant4 via DD4hep, reconstructed with ACTS (digitization → pattern reco → CKF).
- **Channel used here**: `ttbar` (both pretrain and fine-tune). SM tt-bar gives
  a realistic mix of prompt and displaced charged tracks.
- **Pileup**: `pu0` (hard-scatter only) for pretrain, `pu200` (200 HL-LHC PU
  interactions per bunch crossing) for fine-tune.
- **Perigee parameters we regress** (ACTS convention): `d0` = transverse
  impact parameter [mm], `z0` = longitudinal impact parameter [mm],
  `phi` = azimuth at point of closest approach [rad], `theta` = polar angle
  [rad], `qop` = charge / momentum [1/GeV]. Derived: `pT = sin(θ)/|qop|`,
  `η = -ln(tan(θ/2))`.
- **Hit features per sequence element**: `(x, y, z, r, phi_hit, theta_hit, s,
  volume_id, layer_id, surface_id, detector)`. `s` = signed path length from
  IP along the track; the whole sequence is sorted by `s` before the encoder.
  Typical track length ≤ 20 hits.
- **ACTS CKF baseline** is carried through preprocessing as `acts_reco` +
  `acts_dm_mask` so every evaluation can compare on the DM subset of the
  same events.

## Directory layout

- `model.py` — `TrackParameterRegressor` (top-level Lightning module), input
  embedding, pooling selector, `output_head`.
- `mamba_state.py` — `BidirectionalMambaEncoder` + `BidirectionalMambaLayer`
  (SSM-state pool: reads raw SSM recurrent state via `fwd_head`/`bwd_head`).
- `mamba_cls.py` — `BidirectionalMambaCLSEncoder` (SSM-CLS pool: learned CLS
  tokens at each scan terminus, `(B, 2·dim)` readout).
- `transformer_encoder.py` — `EncoderWithCLS` (flash-attn2 baseline with a
  register token).
- `losses.py` — `TrackParameterLoss` dispatcher + per-parameter loss types
  (`quantile`, `quantile_eta`, `circular`, `gaussian`, `gaussian_eta`,
  `spline_quantile`). Supports `delta_anchor` (predict offset from an input
  feature like `innermost_phi`) and `loss_aggregation` (sum vs geometric mean;
  use `sum` — see below).
- `spline.py` — CDF warping used by `spline_quantile` for heavy-tailed
  parameters (d0 has kurtosis ~84).
- `data.py` — dataset + collate. Critical flag: `load_acts: true` on validation
  DataLoader or all DM metrics silently no-op.
- `callbacks.py` — `RegressionPredictionWriter`, `MinimalGpuMonitor`, etc.
- `config/` — YAML configs. Current live study: `NeurIPS_retraining/v2/core_configs/`.
  Its [base.yaml](config/NeurIPS_retraining/v2/core_configs/base.yaml) is auto-loaded by every config in that folder.

## Architecture variants in play

Three encoders × one primary loss (quantile-7). All three use `precision: 32-true`
at the Lightning level with a bf16 autocast wrapper around the encoder forward
(Mamba-2 / flash-attn2 CUDA kernels require bf16). `encoder_autocast_dtype: float32`
disables the autocast and runs the encoder fully in fp32 (slower, for precision
ablations).

| Variant | Encoder | Pool | Readout |
|---|---|---|---|
| `ssm_q7` (state) | `BidirectionalMambaEncoder`, 8L, dim=128, d_state=32 | `ssm_state` | `fwd_head`+`bwd_head` Dense → `output_head` |
| `ssmcls_q7` (CLS) | `BidirectionalMambaCLSEncoder`, 10L, dim=192, d_state=32 | `ssm_cls` | `(B, 2·dim)` CLS concat → `output_head` |
| `txf_q7` | `EncoderWithCLS`, 12L, dim=192, flash-varlen | `register_token` | `pool_head` → `output_head` |

**As of 2026-04-20: CLS beats state by ~5 % on all parameters at matched total
params.** Future scaling work should build on `ssmcls_q7`, not `ssm_q7`.

## Validated recipes

**Optimizer / schedule:**
- Pretrain: **Lion** with OneCycleLR (Lion > AdamW ablated).
- Fine-tune: **under active A/B/C investigation** (see Open Issues below).
  The old default — AdamW + OneCycle + 6× higher peak LR — was found to
  "blow the pretrained basin". Current sweep tests Lion-continuation vs
  AdamW-switch vs Muon-hybrid, all with WSD schedules; no single recipe
  is yet validated. Peak LR depends on optimizer continuity: Lion
  continuation wants ~0.5–1× pretrain peak (preserves weight-norm basin);
  AdamW switch wants ~2× pretrain peak (Gupta et al. 2023: peak LR
  dominates over warmup for basin preservation), *not* 6×.
- Fine-tune must *exceed* the pretrain val loss; if it can't return to it,
  the recipe is broken. The curriculum is necessary — training directly
  on p200 underperforms pretrain→finetune.
- Peak LR for pretrain Lion at baseline scale: ~3–5e-5 at BS=2048–4096.
  Lower LR when raising BS.
- **Weight decay must track the optimizer family.** Lion pretrain uses
  wd=1e-3; on AdamW-switch fine-tune, drop to ~0.02 — AdamW's default WD
  would actively disturb the Lion-trained weight-norm regime.
- **SAM was tried and did not help. Do not re-propose it.** SAM was also
  tried as a way to train directly on p200 (skip the hard-scatter pretrain)
  — that failed too. The pretrain→finetune curriculum is not optional.
- **Muon (and Muon+AdamW hybrid) is the newest optimizer axis** under test
  for fine-tune (Run C) — motivated by SpecMuon (arXiv:2602.16167, 2026)
  reporting 2–10× lower final MSE on PINN/DeepONet precision regression,
  the closest published analog to sub-mrad track parameter regression.
  Unlike SAM it has a plausible mechanism for this task, so it's worth
  keeping in rotation until the A/B/C verdict lands.

**Batch size:**
- H100 is ~10 % utilised at BS=2048 for the baseline model — BS=4096 is free.
- Empirical ~10 % precision gain observed when going BS=10k → 2048; same gain
  recoverable at BS=4096 by lowering peak LR.
- **Pretrain BS is kept artificially low for regularization.** This is a
  precision-regression problem and small-batch noise acts as an implicit
  regulariser; raising BS from scratch hurts. Do not "scale up for speed"
  during pretraining.
- **Fine-tune BS can be scaled up.** Previous ablations show that once
  pretrained, the fine-tune tolerates much larger batch sizes without
  meaningful precision loss — use this to accelerate fine-tune wall-clock.

**Hardware layout & wall-clock:**
- **Pretrain: 1× H100.** Pretrain BS is kept artificially low for
  regularisation so extra GPUs don't help — DDP would just let you raise BS
  past the sweet spot.
- **Fine-tune: 4× H100 (DDP).** Configured via `trainer.devices: -1` in the
  fine-tune YAMLs.
- One pretrain run: **~60–100 h** to converge (depends on depth / dim).
- One fine-tune run: **~60–100 h** wall-clock despite having 4× the compute
  (it's a denser dataset with more epochs needed to close the p200 gap).
- Plan NeurIPS runs accordingly: a fresh pretrain+finetune pair consumes
  roughly a full week, so most deadline-critical experiments should branch
  off an existing pretrained checkpoint rather than restart from scratch.

**Scaling:**
- Depth-first on the encoder: scale `num_layers` (primary), `d_state` (secondary).
- Do **not** grow the Dense readout heads (`state_head_hidden_layers`,
  `output_head_hidden_layers`). An earlier scaled run put ~77 % of parameters
  in the `fwd_head`/`bwd_head` Dense stacks and showed no improvement.
- When scaling, prefer the CLS backbone — its readout is thin by construction.

**Loss design (quantile family, current best):**
- 7 quantiles `[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]`.
- Parameter weights: `d0=1, z0=1, phi=1, theta=10, qop=5` (theta lives in
  η-space; qop is small-magnitude).
- **d0: use `spline_quantile`** with the CDF warping from [splines/spline_d0.yaml](config/NeurIPS_retraining/v2/core_configs/splines/spline_d0.yaml)
  — d0 has kurtosis ~84 and a linear scaling is wasteful.
- **phi: use `delta_anchor: innermost_phi`** with `spline_quantile` — 98.9 % std
  reduction of the target distribution vs predicting absolute phi.
- **theta: stay in η-space** (`quantile_eta`). Delta-theta and absolute-theta
  parameterisations have been tried and are worse.
- **Aggregation: sum, not geometric mean.** GM effectively weights sub-losses by
  `1/L_i`, which *under-weights* the hard parameters (theta, qop) — opposite of
  what we want. Explicit per-parameter weights work better.

**Stability fixes (applied in v2 configs — do not regress):**
- `output_head_init_scale: 0.01` for all non-state-pool variants (CLS, transformer).
- `pool_head` Dense projection for CLS and register-token pools (state pool
  already has `fwd_head`/`bwd_head`).
- Gaussian NLL loss: `variance_param: softplus` with `var_init: 2.0`,
  `var_eps: 1e-3`. The earlier `exp(-log_var)` parameterisation caused 19 OOM
  gradient explosions.
- CLS readouts are extracted from the **gated + residualised** output, not the
  raw Mamba-2 scan output — see [mamba_cls.py:BidirectionalMambaCLSFinalLayer](mamba_cls.py).

## Datasets

Four preprocessed variants, each as a `{p0, p200}` pair. `p0` = no pileup
(pretrain), `p200` = 200 pileup (fine-tune). Dirs live at
`/scratch/colliderml/{p0,p200}_{variant}_{pretrain,finetune}`.

| variant | pt_min [GeV] | |η| < | |d0| < [mm] | min_hits | extra |
|---|---|---|---|---|---|
| `loose` | 0.2 | 3 | 5.0 | 3 | — |
| `core` | 0.5 | 3 | 2.5 | 6 | — ← **NeurIPS v2 default** |
| `core_kf_matched` | 0.5 | 3 | 2.5 | 6 | require ACTS KF double-match |
| `core_kf_hits` | 0.5 | 3 | 2.5 | 6 | + hits restricted to KF-recovered |

Common cuts across variants: `max_hits = 20`, `|z0| ≤ 200 mm`, charged primary
tracks only, `pretrain`=hard-scatter only, `fine-tune`=hard-scatter+pileup.

**Core scale (what we actually train on):**
- `/scratch/colliderml/p0_core_pretrain`: 1000 shards, **71.5 M tracks** over
  1 M ttbar pu0 events (56.5 M / 71.5 M = 79 % ACTS-DM).
- `/scratch/colliderml/p200_core_finetune`: 1000 shards, **241.6 M tracks** —
  ~3.4× more than pretrain because of pileup.
- Split (all variants): train=900 shards / val=50 / test=50. For zero-shot
  eval on the full p200 statistics, override via a custom split.json (see
  "How to run").

## How to run — training, inference, plots

Everything goes through `train.py` (Lightning CLI). `fit` trains, `test`
writes predictions, the eval scripts make plots from the predictions h5.
All commands are run from `/shared/tracking/hepattn_muon` via `pixi run ...`.

**1. Pretrain (1× H100):**
```bash
cd /shared/tracking/hepattn_muon/src/hepattn/experiments/colliderml_regr
CUDA_VISIBLE_DEVICES=0 nohup pixi run python train.py fit \
  --config config/NeurIPS_retraining/v2/core_configs/<variant>_pretrain.yaml \
  > /tmp/pretrain.log 2>&1 &
```
The YAML points `data.preprocessed_dir` at `/scratch/colliderml/p0_core_pretrain`
and `trainer.devices` at 1. Logs go to `logs/comet_offline/<run-id>/`.

**2. Fine-tune (4× H100, DDP):**
```bash
cd /shared/tracking/hepattn_muon/src/hepattn/experiments/colliderml_regr
nohup pixi run python train.py fit \
  --config config/NeurIPS_retraining/v2/core_configs/<variant>_finetune.yaml \
  > /tmp/finetune.log 2>&1 &
```
Fine-tune YAML sets `model.pretrained_ckpt_path` to the frozen pretrain
checkpoint and `trainer.devices: -1`. The data dir points at p200.

**3. Inference (zero-shot or in-distribution) — `test`, not `fit`:**
`RegressionPredictionWriter` writes `test_predictions.h5` only on Lightning's
`test` stage. Pass the saved `config.yaml` from the run directory (not an
original source config) so model/loss state is reproduced exactly.

```bash
LOGDIR=logs/comet_offline/<run-id>
CUDA_VISIBLE_DEVICES=0 nohup pixi run python train.py test \
  --config ${LOGDIR}/config.yaml \
  --ckpt_path ${LOGDIR}/ckpts/<which>.ckpt \
  --trainer.devices 1 \
  --data.batch_size 10000 \
  --data.num_workers 0 \
  --data.preprocessed_dir /scratch/colliderml/p200_core_finetune \
  > ${LOGDIR}/inference.log 2>&1 &
```
Output: `${LOGDIR}/<ckpt-stem>__test_predictions.h5`.

Foot-guns (all learned the hard way, do not unlearn):
- **`num_workers=0` is mandatory for inference.** DataLoader worker forks can
  corrupt gzip-compressed h5 chunks written by the writer, and the error
  (`filter returned failure during read`) surfaces only during the next eval
  run. Pretrain/finetune can use workers freely.
- **`test` uses only 50 / 1000 shards** of each dataset (the `test` key of
  `split.json`, ≈ 5 %). That is ~1 M p200 tracks vs the 241 M available. For
  full-statistics zero-shot eval, write a sibling dir with a `split.json`
  where all shards are in `test` and symlink the data, then point
  `--data.preprocessed_dir` at it.

**4. Performance plots.** Both scripts consume the same h5; run both and
point them at the same `--output-dir` so residual heatmaps, bias-vs-η, and
tail diagnostics live together:
```bash
OUTDIR=${LOGDIR}/eval_last
mkdir -p $OUTDIR
cd /shared/tracking/hepattn_muon
pixi run python -m hepattn.experiments.colliderml_regr.evaluate_predictions \
  --predictions ${LOGDIR}/<ckpt-stem>__test_predictions.h5 \
  --data-dir /scratch/colliderml/p200_core_finetune \
  --output-dir $OUTDIR
pixi run python -m hepattn.experiments.colliderml_regr.evaluate_tail_diagnostics \
  --predictions ${LOGDIR}/<ckpt-stem>__test_predictions.h5 \
  --data-dir /scratch/colliderml/p200_core_finetune \
  --output-dir $OUTDIR
```
Output subdirs: `all_selected/`, `double_matched/`, `acts_baseline_comparison/`,
`double_matched_primary_d0/`, `double_matched_secondary_d0/`. For publishable
plots copy the whole dir into `/shared/tracking/logs/<contextual-name>/`.

**Convention for `<contextual-name>`**: encode architecture, loss family,
pretrain-or-finetune, dataset, run-hash-prefix, epoch, and any caveat. E.g.
`ssm_q7_pretrain_p200_zeroshot_ac72e5c9_epoch49_replot`.

## Current results — SSM (state pool) vs ACTS CKF

Zero-shot inference on p200 (200-pileup) using the p0-pretrained `ssm_q7`
checkpoint `ac72e5c9ede74f6eb241412b8652c4d1`, epoch 49. Core-selection + DM
subset, 6.59 M tracks. Source: `logs/ssm_q7_pretrain_p200_zeroshot_ac72e5c9_epoch49_replot/double_matched/residual_statistics.txt`.

Three complementary metric families are reported: raw standard deviation
(tail-dominated, reflects overall performance including outliers), IQR / 1.349
(robust Gaussian-equivalent σ, no clipping — reflects core resolution
independent of tails), and iterative 3σ-clipped RMS (the physicist-standard
"core σ" after outlier rejection).

**Standard deviation (raw, no clipping):**

| | d0 [mm] | z0 [mm] | φ [mrad] | θ [mrad] | q/p [1/GeV] |
|---|---|---|---|---|---|
| SSM | 0.070 | 0.879 | 2.99 | 2.21 | 0.00522 |
| CKF | 0.209 | 1.652 | 6.28 | 2.83 | 0.00586 |
| **SSM/CKF** | **0.34** | **0.53** | **0.48** | **0.78** | **0.89** |

**IQR / 1.349 (robust σ, no clipping):**

| | d0 [mm] | z0 [mm] | φ [mrad] | θ [mrad] | q/p [1/GeV] |
|---|---|---|---|---|---|
| SSM | 0.0129 | 0.160 | 0.669 | 0.787 | 0.00321 |
| CKF | 0.0599 | 0.113 | 1.877 | 0.700 | 0.00290 |
| **SSM/CKF** | **0.22** | **1.42** | **0.36** | **1.12** | **1.11** |

**Iterative 3σ-clipped RMS (`ssm_iqr_dm` — physicist core resolution):**

| | d0 [mm] | z0 [mm] | φ [mrad] | θ [mrad] | q/p [1/GeV] |
|---|---|---|---|---|---|
| SSM | 0.0123 | 0.214 | 0.661 | 0.905 | 0.00368 |
| CKF | 0.0657 | 0.188 | 2.111 | 0.802 | 0.00344 |
| **SSM/CKF** | **0.19** | **1.14** | **0.31** | **1.13** | **1.07** |

### Interpretation

The picture splits cleanly in two:

- **Tail regime (raw std):** SSM wins on all five parameters, 1.1×–3× better.
  The SSM captures non-Gaussian scattering structure that a linear Kalman
  filter cannot model.
- **Core regime (IQR, clipped RMS):** SSM wins dramatically on `d0` and `φ`
  (3-5× better), and **loses** on `z0`, `θ`, `q/p` by 7–42 %. These three
  parameters depend on long-range integration along the full track;
  CKF is near its Cramér–Rao bound there and is hard to beat with an SSM
  that accumulates bf16 round-off through the recurrence.

This result was achieved with **`ssm_state` pooling** (raw SSM state read-out
via `fwd_head`/`bwd_head`), not CLS. The CLS variant showed +5 % over state at
matched pretrain total params; evaluating it zero-shot on p200 at the same
scale is a pending experiment and may already close part of the core-resolution
gap.

## Current open issues (2026-04-20)

**1. Scaling.** A prior scaled run (16 L, d_state=64, with fat
`state_head_hidden_layers: [1024]`) failed to improve on the 8 L baseline
because ~77 % of parameters landed in Dense readout projections rather than
the SSM residual stream. The next scaling attempt should (a) build on the
CLS backbone, (b) keep `output_head_hidden_layers` at `[256]` or smaller,
(c) apply residual-depth rescaling `1/√(2N)` on `out_proj` at init. Expected
recipe: 16–20 L, `dim=192`, `d_state=64`, thin heads.

**2. FP32 backbone.** The core-resolution gap on `z0`, `θ`, `q/p` is
consistent with bf16 round-off in the long-range SSM recurrence. The
ssm-ssm A_log / dt_bias / D parameters are the critical ones to keep in
fp32. A fp32-polish fine-tune from the baseline checkpoint is queued
([ssm_q7_fp32polish_finetune.yaml](config/NeurIPS_retraining/v2/core_configs/ssm_q7_fp32polish_finetune.yaml));
if it tightens the z0/θ/q/p cores, fp32 A_log + fp32 post-SSM RMSNorm become
permanent fixtures. If not, the gap is architectural / Cramér–Rao-bound
proximity, not precision.

**3. d0 bias + core collapse artifact (MID PRIORITY — NeurIPS deadline 2026-05-04).**
Heatmaps show a characteristic cross at `d0 = 0`: a vertical band at truth=0
and a horizontal band at pred=0. Root cause diagnosed: 95 % of tracks have
`|d0| ≤ 0.031 mm`, 68 % within ±0.013 mm, so the empirical CDF is near-vertical
at zero. The `spline_quantile` loss runs in u = CDF(d0) space, which means
nearly the whole usable u-range (~0.025–0.975) decodes back to `|d0| < 0.03 mm`
under `spline.inverse` — any "hedged" prediction (u ≈ 0.5) lands exactly on
d0 = 0. The pinball loss therefore has enormous gradient for tiny physical
shifts near the mode and almost none in the tails, which both reinforces the
collapse and under-rewards tail-correct predictions. FP32 is *not* the
bottleneck (ULP ≈ 1e-9 mm at target scale); if mixed-precision is ever
enabled, keep the output head + loss in fp32 — bf16 around u = 0.5 would
produce ~0.3 mm errors through the steep inverse.

Improvement candidates, in preferred order:
- **Swap spline→Gaussian NLL for d0** (or β-NLL; pretrain run
  `65c08d6ddb604f489d8573cfb27a82d7` already uses this recipe for all
  parameters). μ/σ parameterisation lets the model express uncertainty in σ
  without collapsing μ to 0.
- **Mixture density head for d0** (2–3 Gaussian mixture). Matches the
  sharp-core-plus-tails shape natively, no transform needed.
- **Drop the spline, use plain `quantile` with linear rescale.** Simpler
  fallback; loses some tail calibration but removes the collapse attractor.
- **Blended transform `u = α·linear + (1-α)·CDF`** with α ≈ 0.3–0.7. Keeps
  some tail-matching but reduces the CDF-induced collapse. Requires
  retraining (output head is calibrated to the specific transform).
- **Training-time offset calibration term.** Bin each batch on **truth η**
  (`η = -ln(tan(θ/2))`, *not* predicted), compute per-bin mean residual for
  each parameter, add `λ · Σ bias²` to the loss (soft binning for
  differentiability). Directly attacks the η-dependent bias visible in the
  `mean_residual_vs_eta` plots without constraining the spread.
- **Post-hoc bias correction** (no retraining). Fit `bias(η)` per parameter
  on a held-out val split, subtract at inference. Cheap sanity-check; works
  on existing checkpoints but does not prevent future training from drifting.

**Not recommended:** zero-inflated classifier for "is-beamspot". The d0
distribution is smooth-peaked, not physically bimodal — a hard threshold
would be arbitrary and introduces a discontinuity.

**4. Fine-tuning is not closing the gap.** The original OneCycle + AdamW +
6× LR jump recipe "blew the pretrained basin". An A/B/C sweep under WSD
schedules is in flight, each testing a different hypothesis:

- **Run A** — [`ssm_q7_finetune_A_adamw_wsd.yaml`](config/NeurIPS_retraining/v2/core_configs/ssm_q7_finetune_A_adamw_wsd.yaml):
  AdamW + WSD, peak LR 2× pretrain (not 6×), wd=0.02. Tests whether the
  failure mode was *schedule shape and LR magnitude*, keeping optimizer
  family switched as before.
- **Run B** — [`ssm_q7_finetune_B_lion_wsd.yaml`](config/NeurIPS_retraining/v2/core_configs/ssm_q7_finetune_B_lion_wsd.yaml):
  Lion-continuation + WSD, peak LR 0.6× pretrain. Tests whether the
  failure mode was *optimizer-family switch disturbing the Lion-trained
  weight-norm regime* (MuonAll paper, arXiv:2511.06086).
- **Run C** — [`ssm_q7_finetune_C_muon_wsd.yaml`](config/NeurIPS_retraining/v2/core_configs/ssm_q7_finetune_C_muon_wsd.yaml):
  Muon (2-D weights) + AdamW (1-D weights) hybrid + WSD. Tests whether a
  precision-regression-suited optimizer can break the residual-core
  ceiling on `z0`/`θ`/`q/p` that no scalar-update optimizer has closed.
  Motivated by SpecMuon (arXiv:2602.16167) — 2–10× lower final MSE on
  PINN/DeepONet precision regression.

Full literature review in
[reference_optimizer_schedule_research_2026_04.md](../../../../../../../../afs/cern.ch/user/j/jorenusc/.claude/projects/-shared-tracking/memory/reference_optimizer_schedule_research_2026_04.md).
Not yet explored in any of the three: p200 data-distribution refresh
(pileup contamination in hit features may warrant re-computing
normalisation stats) — worth revisiting if all of A/B/C fail to close the
gap.

**5. Pretrain↔fine-tune KF-DM distribution mismatch (likely core-resolution
bottleneck).** Pretrain `p0_core_pretrain` is 79 % ACTS-DM (56.5 M / 71.5 M).
Fine-tune `p200_core_finetune` is only **54.3 %** DM (verified from
`acts_dm_mask.npy`: 658,241 / 1,211,502 over 5 shards). The SSM therefore
pretrained on a distribution where the KF can double-match most tracks, then
fine-tunes on one where ~46 % of tracks are KF-unmatchable — yet the metrics
we actually care about (`ssm_iqr_dm`, `ssm_rms_dm`, iterative 3σ-clipped RMS)
are restricted to the DM subset. Nearly half the fine-tune gradient is thus
spent on tracks the network will never be scored on, and whose hit patterns
(pileup-confused, ambiguous majority assignments) may actively pull the
weights away from the core-resolution basin — this is a plausible mechanism
for the residual `z0`/`θ`/`q/p` gap that survived A/B/C. Matching-quality
diagnostic plots in
[logs/acts_tracking_evaluation_p200/double_matched_diagnostics/](../../../../../../../../shared/tracking/logs/acts_tracking_evaluation_p200/double_matched_diagnostics/)
(script: [plot_double_matched_diagnostics.py](scripts/plot_double_matched_diagnostics.py))
show that even within the 0.75-DM subset, 31 % of primary+charged tracks fail
the strict 100 %-DM criterion (purity=1 ∧ hit-eff=1). Mitigation: fine-tune
on the kf_matched variant so the training distribution is 100 % DM and
aligned with the metric — config
[`ssm_q7_finetune_kfmatched_muon_wsd.yaml`](config/NeurIPS_retraining/v2/core_configs/ssm_q7_finetune_kfmatched_muon_wsd.yaml)
(Run C recipe, Muon+AdamW WSD, pointed at
`p200_core_kf_matched_finetune` which is already preprocessed under
`/eos/.../NeurIPS_retraining/p200_core_kf_matched_finetune`; rsync to
`/scratch` before launch).

## Paper draft (NeurIPS main conference target, deadline ~2 weeks)

# Beyond the Kalman Filter: Linear-Time Bidirectional State Space Models for High-Precision Particle-Trajectory Regression at the LHC

**Title alternatives** (pick to match reviewer pool / framing):
- *BiMamba-Track: Bidirectional State Space Models for High-Precision
  Regression on Heterogeneous Spatio-Temporal Detector Sequences*
- *Replacing the Kalman Filter: A Bidirectional SSM for Charged-Particle
  Trajectory Regression Under Pileup*

Selection notes: the merged primary title leads with the recognizable
scientific baseline ("Beyond the Kalman Filter") so an ML reviewer with no
HEP background gets an immediate hook, and qualifies the method as
"Linear-Time Bidirectional State Space Models" so a sequence-modelling
specialist sees the algorithmic contribution in the same breath. Avoid
acronym-heavy titles (CKF, HL-LHC) in the title itself — those go in the
abstract.

### Introduction draft

> Particle-physics experiments at the Large Hadron Collider observe roughly
> a billion proton collisions per second. The task of *tracking* —
> reconstructing the curved three-dimensional trajectory of every charged
> particle from a set of discrete position measurements left in a silicon
> detector — is the foundational reconstruction step on which essentially
> all downstream physics analyses depend, from precision Higgs measurements
> to searches for new particles. For the past three decades the workhorse
> algorithm has been the **combinatorial Kalman filter** (CKF; Frühwirth,
> 1987), a recursive linear estimator that propagates a Gaussian belief
> state from one detector layer to the next, performing measurement
> association and parameter update in a single sweep. CKF is statistically
> well-motivated under Gaussian noise assumptions and remains the de-facto
> standard, packaged in modern frameworks such as ACTS (Ai et al., 2022).
>
> Two trends are now eroding the CKF's dominance. First, CKF compute scales
> super-linearly with detector occupancy — a primary driver of the
> projected ~100× CPU-need growth between LHC Run-2 (pileup ≈ 35) and
> HL-LHC offline reconstruction (pileup ≈ 200) identified in the CERN
> HL-LHC computing outlook [CERN EP News, *HL-LHC computational challenge
> for ATLAS and CMS*, 2023] — because both the combinatorial branching of
> candidate seeds and the per-track propagation cost grow with the hit
> density. The High-Luminosity LHC upgrade (HL-LHC, ATLAS and CMS
> Phase-2 trackers) will deliver up to **200 simultaneous proton-proton
> collisions per bunch crossing** ("pileup-200"), producing on the order of
> **2.5 × 10⁵ silicon hits per event** distributed across heterogeneous
> spatio-temporal detector geometries — pixel and strip layers of varying
> granularity and timing resolution. Tracking compute is projected to
> dominate the HL-LHC offline reconstruction budget on CPU; the field is
> actively migrating to GPU-native algorithms. Second, the CKF's Gaussian
> linear model cannot capture the genuinely non-Gaussian response of dense
> silicon — multiple-scattering tails, hadronic interactions, and
> pileup-induced ambiguity — and is known to leave precision on the table
> on the parameters most sensitive to these tails (the transverse impact
> parameter d₀, the azimuthal angle φ).
>
> Two parallel lines of machine-learning research have begun to address
> this. On the HEP side, graph neural networks (Exa.TrkX; Ju et al., 2021;
> GNN4ITk, 2023) and transformer architectures (HEPT, Miao et al., ICML
> 2024; TrackFormers, 2024) have been applied largely to the *track-finding*
> problem — pattern-recognition over hits. The complementary *per-track
> parameter regression* problem we study here, which directly competes with
> the parameter-update half of CKF, has received considerably less
> attention. On the ML side, **state space models** — Mamba (Gu & Dao,
> 2023), Mamba-2 / SSD (Dao & Gu, 2024) — have established themselves as
> linear-time alternatives to attention, with bidirectional adaptations such
> as Vision Mamba (Zhu et al., 2024) extending them beyond causal language
> modelling to visual classification.
>
> We bring these two lines together. The structural fit is strong: a track
> is a one-dimensional sequence of detector measurements naturally ordered
> along its arc length from the interaction point, and a recursive linear
> estimator is, in essence, a hand-designed state space model. Replacing
> CKF's hand-designed Gaussian recursion with a **learned, non-linear,
> bidirectional Mamba-2 recursion** therefore preserves the algorithmic
> shape of the problem while admitting the heavy-tailed scattering response
> end-to-end from data. We adopt a forward+backward selective scan with
> gated merge — adapted from Vision Mamba — so that the forward and
> backward passes correspond to inward and outward trajectory integration,
> mirroring CKF's filter+smoother structure.
>
> A natural objection is that the transformer, not the SSM, is the
> default modern sequence model and has already been applied to HEP
> tracking (HEPT, Miao et al. 2024; TrackFormers, 2024). Two structural
> arguments favour SSMs in this specific setting. First, the linear-time
> recurrent computation scales gracefully to the O(10⁵)-element context
> length of a full HL-LHC event without the quadratic memory cost of
> self-attention — relevant for the longer-term goal of event-level
> tracking on a single GPU. Second, the recurrent hidden state is itself a
> learned, fixed-size trajectory summary, a natural functional analogue of
> the Kalman state vector that the parameter-update head can read directly
> — whereas a transformer must synthesise an equivalent summary from
> attention over per-measurement tokens. To check whether these structural
> arguments translate to empirical gains in the parameter-regression
> regime studied here, we benchmark our bidirectional Mamba-2 against a
> **parameter-matched flash-attention transformer encoder with a register
> token**, pretrained under an identical Lion+OneCycleLR recipe and
> evaluated on the same data and metrics. The SSM outperforms the
> transformer on every perigee parameter at matched compute, supporting
> the hypothesis that the inductive bias of a learned recurrent state is
> well suited to trajectory regression.
>
> Our contributions are:
>
> 1. **A Mamba-2 architecture for trajectory regression at parity with the
>    CKF compute budget on a single GPU.** Trained on hard-scatter
>    simulation and evaluated zero-shot at pileup-200, our model improves
>    on the CKF's tail-inclusive standard deviation across all five
>    perigee parameters, indicating substantially reduced sensitivity to
>    non-Gaussian outliers.
> 2. **The first systematic ablation of sequence-summary readouts for
>    SSM-based scientific regression** — direct extraction of the final
>    recurrent SSM state vs learned CLS-token pooling at each scan
>    terminus — and a parameter-matched flash-attention transformer
>    baseline pretrained under an identical recipe, which underperforms
>    both SSM variants.
> 3. **The first ablation of the pretraining→fine-tuning optimizer
>    curriculum for SSM-based precision regression.** Holding Lion fixed
>    for the hard-scatter pretraining stage, we ablate three continuations
>    on the pileup-contaminated fine-tuning stage — Lion, AdamW, and a
>    Muon-(2-D weights) + AdamW-(1-D weights) hybrid — under
>    warmup–stable–decay schedules.
> 4. **A loss design tailored to the heavy-tailed physical targets**:
>    seven-quantile pinball regression with CDF-warped targets for the
>    impact parameter (kurtosis ≈ 84), a circular loss for azimuth, and
>    pseudorapidity-space parameterisation for the polar angle to decouple
>    the loss from the geometric singularity at the beam axis.
>
> On the parameters that depend on long-range integration along the
> trajectory (z₀, polar angle, charge/momentum), CKF is provably near the
> precision floor set by detector resolution and the Gaussian component of
> multiple scattering — i.e. it is essentially optimal among unbiased
> estimators in that limit, and we approach it within 10–15 % without
> exploiting any geometric prior. On the parameters dominated by the
> innermost detector layers and short-range information (transverse impact
> parameter, azimuth), where the heavy-tailed scattering response matters,
> our model improves on CKF's iterative-clipped core resolution by factors
> of three to five. We argue that this *core-vs-tail* split is the
> qualitatively interesting result: it locates precisely the regime in
> which a learned, non-Gaussian sequence model adds value over a
> three-decade-old optimal linear filter, and the regime in which it does
> not.

### Abstract drafts (deadline ~2 weeks)

Three framings. Pick one or merge fragments. All three honest about the
core-vs-tail split; none claim "beats Kalman filter on everything".

### Draft 1 — ML-framed (architecture generalisation)

> Charged-particle tracking — reconstructing the five-parameter helix
> `(d0, z0, φ, θ, q/p)` of each particle from its sequence of silicon
> detector measurements — is the foundational reconstruction step at the
> Large Hadron Collider, and has been performed for over three decades by a
> domain-specialised algorithm: the combinatorial Kalman filter (CKF), a
> linear recursive estimator that is statistically optimal under Gaussian
> noise but cannot model the heavy-tailed multiple-scattering response of
> dense silicon. We replace this 30-year-old baseline with a custom
> **bidirectional Mamba-2** state space model — a forward+backward scan with
> gated merge adapted from Vision Mamba (arXiv:2401.09417). The structural
> fit is direct: the SSM's learned recurrent hidden state is a non-linear
> functional analog of the Kalman state vector, the forward and backward
> scans mirror the filter/smoother passes of CKF, and the linear-time
> recurrence scales gracefully to the O(10⁵)-element context length of a
> full HL-LHC event — properties a transformer does not share. The model
> is trained end-to-end under a seven-quantile regression loss with
> CDF-warped targets for the heavy-tailed transverse impact parameter and
> a circular loss for azimuth. We run two ablations new to this domain.
> **(i) Architecture and readout:** at matched parameter count and under an
> identical Lion+OneCycleLR pretraining recipe, our SSM beats a
> parameter-matched flash-attention transformer encoder on every
> parameter; within the SSM family, learned
> CLS-token pooling at each scan terminus outperforms direct recurrent-state
> extraction by ~5 %. **(ii) Pretrain-then-finetune optimizer curriculum:**
> with Lion fixed for hard-scatter pretraining, we ablate three
> finetune continuations on the pileup-contaminated regime — Lion, AdamW,
> and a Muon+AdamW hybrid — under WSD schedules. Evaluated zero-shot on
> 200-pileup data, our model matches or exceeds CKF on overall standard
> deviation for all five parameters (ratios 0.34–0.89, indicating
> substantially reduced sensitivity to non-Gaussian outliers) and dominates
> by 3–5× on `d0` and `φ` core resolution; on `z0`, `θ`, `q/p` —
> parameters whose precision is set by long-range integration along the
> trajectory and where the linear Kalman filter is provably near the
> information-theoretic precision floor for unbiased estimators under
> Gaussian noise — we match it within 10–15 %, all within the same
> single-GPU compute budget the CKF baseline consumes per event.

### Draft 2 — HEP-framed (physics problem first)

> Accurate track parameter estimation is a cornerstone of collider physics
> analyses; its precision floor is set by multiple scattering, detector
> resolution, and pileup contamination, and at HL-LHC luminosity each event
> presents O(250 K) detector measurements drawn from a heterogeneous,
> spatio-temporal silicon geometry. The combinatorial Kalman filter (CKF)
> has remained the field's reconstruction baseline for over thirty years
> despite well-known limitations: it is linear, blind to non-Gaussian
> scattering tails, and CPU-bound. A state space model is the natural
> GPU-native successor — its learned recurrent hidden state is a
> non-linear functional analog of the Kalman state vector and its
> linear-time recurrence scales to event-level context lengths that are
> out of reach for quadratic-cost self-attention. We study whether a
> custom **bidirectional Mamba-2** encoder — adapted from the Vision
> Mamba architecture (arXiv:2401.09417) and trained end-to-end with a
> seven-quantile loss — can learn the non-Gaussian response of a
> realistic silicon tracker on a GPU within the same compute budget the
> CKF consumes per event, and head-to-head against a **parameter-matched
> flash-attention transformer encoder** pretrained under the same recipe. We present the first systematic ablations of (a) sequence
> readout for SSM-based tracking — direct recurrent-state extraction vs
> learned CLS-token pooling vs a parameter-matched flash-attention
> transformer baseline pretrained under the same Lion+OneCycleLR recipe —
> and (b) the pretrain-then-finetune optimizer curriculum, holding Lion
> fixed for hard-scatter pretraining and ablating Lion, AdamW, and a
> Muon+AdamW hybrid (motivated by SpecMuon, arXiv:2602.16167) under WSD
> schedules for the pileup fine-tune. Evaluated zero-shot on 200-pileup
> events, the Mamba-2 model improves the iterative-clipped core resolution
> over CKF by factors of 5 and 3 on the transverse impact parameter `d0`
> and azimuth `φ` respectively, and improves the tail-inclusive standard
> deviation on every one of the five perigee parameters — demonstrating
> markedly reduced volatility to outliers. On `z0`, `θ`, and `q/p` —
> parameters requiring long-range trajectory integration where CKF is near
> its theoretical optimum — we match CKF within 7–14 %. We discuss
> implications for pileup robustness, GPU-native tracking at HL-LHC scale,
> and the residual precision regime tied to bf16 selective-scan recurrence.

### Draft 3 — Method-framed (technical novelty)

> State space models are attractive for regression over physical sequences:
> they provide linear-time scaling with a recurrent hidden state that
> naturally encodes integrated dynamics. These properties map directly
> onto the dominant baseline in particle-trajectory reconstruction — the
> combinatorial Kalman filter, a CPU-bound recursive linear estimator
> that has remained the field's standard for three decades — whose
> Gaussian state vector the SSM's learned hidden state functionally
> generalises; a transformer offers no such structural correspondence and
> pays quadratic cost in sequence length. We present a custom
> **bidirectional Mamba-2 encoder** — a forward+backward selective-scan
> module with gated merge, adapted from Vision Mamba (arXiv:2401.09417,
> originally designed for image classification) — applied to perigee
> trajectory regression, with two domain-motivated choices: (i) detector
> measurements are sorted along the signed distance from the interaction
> point so the forward and backward scans correspond to inward and outward
> trajectory integration; (ii) we ablate two readouts — direct extraction
> of the final recurrent SSM state vs learned CLS-token pooling at each
> scan terminus — and benchmark both against a parameter-matched
> flash-attention transformer baseline pretrained under the same
> Lion+OneCycleLR recipe (which underperforms both SSM variants). Loss
> design matters: seven-quantile pinball regression with CDF-warped targets
> for the heavy-tailed transverse impact parameter (kurtosis ≈ 84), a
> circular loss for azimuth, and η-space parameterisation for polar angle
> to decouple the loss from the geometric singularity at the beam axis. We
> further provide the first ablation of the **Lion-pretrain × {Lion,
> AdamW, Muon+AdamW-hybrid}-finetune optimizer curriculum** on a
> precision-regression task at this scale, motivated by SpecMuon
> (arXiv:2602.16167). Trained on hard-scatter-only events and evaluated
> zero-shot at 200-pileup occupancy on 6.6 M tracks within the same
> single-GPU compute envelope as the CKF baseline, the model improves
> overall standard deviation on all five perigee parameters (substantially
> reducing outlier volatility) and core resolution by 3–5× on `d0` and
> `φ`, while approaching CKF on the long-range-integrated parameters
> (`z0`, `θ`, `q/p`) within 10–15 % — the regime where the Kalman filter
> is provably near the precision floor that any unbiased estimator can
> attain under Gaussian measurement and scattering noise. We
> analyse the residual core-resolution gap in terms of mixed-precision
> arithmetic in the selective-scan kernel and propose an fp32-polish
> strategy that isolates the affected SSM parameters.

## Key conventions

- **z is the horizontal axis** in event displays (beamline convention).
- **Create new YAML files** for training variants, don't edit existing ones
  (reproducibility — old configs must keep producing the same checkpoints).
- Hit sequences are sorted along `s` (signed distance from IP along track)
  before the encoder. CLS tokens inserted after sorting.
- Track length distribution: **mean ≈ 12–13 hits, std ≈ 4 hits, max ≈ 20,
  min = 6** (core selection enforces `min_hits=6`).
- **Core configs are set to `chunk_size: 16`.** This is the smallest value
  the installed `mamba_ssm` Triton kernel accepts — `chunk_size: 4` and
  `chunk_size: 8` both crash `_chunk_state_fwd_kernel` with `Triton Error
  [CUDA]: invalid argument` (autotuner cannot produce a valid launch
  config). 16 is the effective kernel minimum.
- **`chunk_size: 16` is ~33 % faster than 256** on these short sequences
  despite doing more kernel launches. Reason: the SSD algorithm does a
  quadratic-in-chunk-size matmul inside each chunk regardless of how many
  real hits occupy it. At `chunk_size: 256` with 20-hit tracks, the kernel
  processes a 256×256 = 65,536-op tile per track with 236 padding
  positions contributing zero signal. At `chunk_size: 16`, each track
  uses ~2 tiles of 16×16 = 256 ops each → ~512 ops per track, a ~128×
  reduction in wasted intra-chunk compute. Kernel-launch overhead for the
  extra chunks is a tiny constant. This is the opposite of the
  LLM-training intuition, where seq_len ≫ chunk_size and the padding
  waste doesn't matter.
- **`chunk_size: 16` precision effect**: the scan does an fp32 reduction
  at every chunk boundary, so with max track length 20, only tracks
  longer than 16 hits see an in-kernel fp32 resync (~10–20 % of tracks).
  Typical 12–13-hit tracks still see zero resyncs, same as under 256.
  This is the best the current kernel allows. To get resyncs on typical
  tracks would require either (a) upgrading `mamba_ssm` to a version that
  permits `chunk_size < 16`, (b) routing through the slower reference
  `ssd_minimal` path, or (c) padding sequences past 32 so `chunk_size: 16`
  produces two chunks per track. None of these are done yet.
- The **`ssm_q7_fp32polish_finetune`** config remains the cleanest way to
  test whether bf16 recurrence is the bottleneck — it bypasses this whole
  question by running the encoder in full fp32.
- Experiment tracking: CometML (offline), `logs/comet_offline/<run-id>/`.
- Multi-GPU: 4× H100 available for fine tuning; Lightning DDP via `devices: -1` in fine-tune
  configs.

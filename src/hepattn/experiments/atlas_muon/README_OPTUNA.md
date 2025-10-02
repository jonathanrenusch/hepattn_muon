# ATLAS Muon Tracking Hyperparameter Optimization with Optuna

This directory contains scripts for running hyperparameter optimization on the ATLAS muon tracking model using Optuna with multi-GPU support.

## Files

- `optuna_tune.py` - Main hyperparameter optimization script
- `optuna_base_config.yaml` - Base configuration with static parameters
- `test_optuna_setup.py` - Test script to validate the setup
- `run_multi_gpu_optimization.sh` - Example script for multi-GPU optimization
- `analyze_results.py` - Script to analyze optimization results

## Quick Start

1. **Test the setup:**
   ```bash
   cd /shared/tracking/hepattn_muon/src
   python hepattn/experiments/atlas_muon/test_optuna_setup.py
   ```

2. **Run a single GPU optimization:**
   ```bash
   python hepattn/experiments/atlas_muon/optuna_tune.py --gpu 0 --n-trials 50
   ```

3. **Run multi-GPU optimization:**
   ```bash
   # Start multiple processes, each on a different GPU
   python hepattn/experiments/atlas_muon/optuna_tune.py --gpu 0 --n-trials 100 &
   python hepattn/experiments/atlas_muon/optuna_tune.py --gpu 1 --n-trials 100 &
   python hepattn/experiments/atlas_muon/optuna_tune.py --gpu 2 --n-trials 100 &
   wait
   ```

## Hyperparameters Being Optimized

The optimization script tunes the following hyperparameters:

### Model Architecture
- `num_encoder_layers`: {1, 2, 3, 4}
- `num_decoder_layers`: {1, 2, 3, 4} 
- `dim`: {16, 32, 64} - Encoder/decoder embedding dimension

### Task Weights
- `*_cost_weight`: {0.1, 1.0, 10.0} - Cost weights for matching
- `*_loss_weight`: {0.1, 1.0, 10.0} - Loss weights for gradient computation

### Dense Network Architecture
- `*_hidden_dim`: {64, 128, 256, 512} - Hidden layer dimensions for all dense networks

## Objective Function

The optimization minimizes the sum of raw losses from all tasks without their weight multipliers:
- Track validity loss (BCE)
- Track-hit mask loss (BCE)
- Parameter regression loss (Smooth L1)
- Charge classification loss (BCE)

This ensures the optimization focuses on the fundamental model performance rather than the relative task weighting.

## Database and Multi-GPU Coordination

All optimization runs share a SQLite database (`optuna_study.db` by default) that coordinates trials across GPUs:
- Prevents duplicate trials
- Enables warmup period across all GPUs (50 trials default)
- Uses TPE (Tree-structured Parzen Estimator) sampler after warmup

## Command Line Options

```bash
python optuna_tune.py [OPTIONS]

Options:
  --gpu GPU_ID              GPU ID to use (required)
  --n-trials N              Number of trials to run (default: 100)
  --study-name NAME         Optuna study name (default: atlas_muon_optuna_study)
  --db-path PATH            SQLite database path (default: ./optuna_study.db)
  --base-config PATH        Base config file path
  --n-warmup N              Warmup trials for TPE sampler (default: 50)
  --pruner                  Enable pruning of unpromising trials
```

## Training Configuration

Each trial uses:
- **Early stopping**: 5 epochs patience on val/loss
- **Checkpointing**: Save top-1 best model based on val/loss
- **Comet ML logging**: Each trial gets a unique experiment name
- **Max epochs**: 50 (as configured in base config)
- **Precision**: bf16-mixed for faster training

## Example Multi-GPU Workflow

1. **Start optimization on multiple GPUs:**
   ```bash
   # Terminal 1
   python optuna_tune.py --gpu 0 --n-trials 200 --study-name my_optimization
   
   # Terminal 2  
   python optuna_tune.py --gpu 1 --n-trials 200 --study-name my_optimization
   
   # Terminal 3
   python optuna_tune.py --gpu 2 --n-trials 200 --study-name my_optimization
   ```

2. **Monitor progress:**
   ```bash
   # Check database for completed trials
   sqlite3 optuna_study.db "SELECT COUNT(*) as completed_trials FROM trials WHERE state = 'COMPLETE';"
   ```

3. **Analyze results:**
   ```bash
   python analyze_results.py --db-path optuna_study.db --study-name my_optimization
   ```

## Tips

- **Start small**: Begin with `--n-trials 10` to verify everything works
- **Monitor resources**: Watch GPU memory usage, especially with larger `dim` values
- **Use pruning**: Add `--pruner` flag to stop unpromising trials early
- **Backup database**: Copy `optuna_study.db` periodically as backup
- **Check logs**: Each trial creates its own Comet ML experiment for detailed monitoring

## Troubleshooting

1. **Import errors**: Ensure you're in the right conda environment with all dependencies
2. **GPU memory issues**: Reduce batch size in base config or use smaller `dim` values
3. **Database locks**: If you see SQLite lock errors, wait and retry - multiple processes accessing DB simultaneously
4. **Failed trials**: Check Comet ML logs for specific trial failures

## Expected Runtime

- **Single trial**: ~30-60 minutes depending on hyperparameters
- **100 trials on 4 GPUs**: ~6-12 hours total
- **Warmup period**: First 50 trials will use random sampling, then TPE optimizer kicks in
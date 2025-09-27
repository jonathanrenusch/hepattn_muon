# Parallel GPU Inference Script

This script allows you to run multiple ATLAS muon filtering inference jobs in parallel on the same GPU to test parallelization capabilities.

## Usage

From the `/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon` directory:

```bash
# Basic usage - run 3 parallel jobs with monitoring
pixi run python run_parallel_inference.py -n 3

# Run 5 jobs and save outputs to a specific directory
pixi run python run_parallel_inference.py -n 5 --output-dir results

# Run all 10 jobs without interactive monitoring
pixi run python run_parallel_inference.py -n 10 --no-monitor

# Custom monitoring interval (check every 60 seconds)
pixi run python run_parallel_inference.py -n 4 --check-interval 60
```

## Features

- **Parallel Job Submission**: Submits 1-10 jobs simultaneously using different config files
- **Real-time Monitoring**: Tracks job status, CPU usage, and memory consumption
- **Output Collection**: Captures all job outputs in separate files
- **Results Summary**: Generates JSON summary with runtime statistics
- **Graceful Cleanup**: Handles interruptions and cleans up running jobs

## Config Files

The script uses pre-existing config files:
- `atlas_muon_filtering_small_inference0.yaml` through `atlas_muon_filtering_small_inference9.yaml`
- Each config points to a different test data directory (`gut_check_test_data0` through `gut_check_test_data9`)

## Output Files

For each job `N`, the script creates:
- `filtering_inference_output_jobN.out` - Complete job output
- `parallel_inference_summary_TIMESTAMP.json` - Results summary

## Monitoring

When monitoring is enabled (default), the script shows:
- Current job statuses (running/completed)
- Resource usage (CPU %, Memory MB)
- Real-time updates every 30 seconds (configurable)
- Final status report with runtimes and output previews

## Example Output

```
Submitting 3 parallel inference jobs...
Start time: 2025-09-17 16:05:16
--------------------------------------------------
Job 0: Submitting with config .../atlas_muon_filtering_small_inference0.yaml
  → PID: 12345, Output: filtering_inference_output_job0.out
Job 1: Submitting with config .../atlas_muon_filtering_small_inference1.yaml
  → PID: 12346, Output: filtering_inference_output_job1.out
Job 2: Submitting with config .../atlas_muon_filtering_small_inference2.yaml
  → PID: 12347, Output: filtering_inference_output_job2.out

✅ All 3 jobs submitted successfully!
Job PIDs: [12345, 12346, 12347]

🔍 Monitoring jobs every 30 seconds...
Press Ctrl+C to stop monitoring and get final status
```

## Arguments

- `-n, --num-jobs`: Number of parallel jobs (1-10) **[Required]**
- `--config-base-path`: Base path for config files (default: auto-detected)
- `--output-dir`: Directory for output files (default: current directory)
- `--check-interval`: Status check interval in seconds (default: 30)
- `--no-monitor`: Submit jobs and exit without monitoring

## Notes

- Maximum of 10 jobs supported (limited by available config files)
- Jobs run with `nohup` to continue after script termination
- All jobs use the same GPU (device 1 as configured)
- Press Ctrl+C during monitoring to get final status and exit gracefully
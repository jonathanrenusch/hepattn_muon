#!/usr/bin/env python3
"""
Script to run parallel GPU inference jobs for ATLAS muon filtering.
Submits multiple jobs with different config files and monitors their progress.
"""

import argparse
import subprocess
import time
import os
import signal
import psutil
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime


class JobManager:
    """Manages parallel job submission and monitoring."""
    
    def __init__(self, num_jobs: int, config_base_path: str, output_dir: str = "."):
        self.num_jobs = num_jobs
        self.config_base_path = config_base_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Job tracking
        self.jobs: Dict[int, Dict] = {}
        self.start_time = datetime.now()
        
        # Ensure we don't exceed the maximum of 10 jobs
        if num_jobs > 10:
            raise ValueError("Maximum number of jobs is 10 (configs 0-9 available)")
            
    def submit_jobs(self) -> None:
        """Submit all jobs in parallel."""
        print(f"Submitting {self.num_jobs} parallel inference jobs...")
        print(f"Start time: {self.start_time}")
        print("-" * 50)
        
        for job_id in range(self.num_jobs):
            self._submit_single_job(job_id)
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
            
        print(f"\n✅ All {self.num_jobs} jobs submitted successfully!")
        print(f"Job PIDs: {[job['pid'] for job in self.jobs.values()]}")
        
    def _submit_single_job(self, job_id: int) -> None:
        """Submit a single job with the given job_id."""
        config_path = f"{self.config_base_path}{job_id}.yaml"
        output_file = self.output_dir / f"filtering_inference_output_job{job_id}.out"
        
        # Construct the command
        cmd = [
            "nohup", "pixi", "run", "python", "-m", 
            "hepattn.experiments.atlas_muon.run_filtering",
            "test", "-c", config_path
        ]
        
        print(f"Job {job_id}: Submitting with config {config_path}")
        
        # Submit the job
        with open(output_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )
        
        # Store job information
        self.jobs[job_id] = {
            'pid': process.pid,
            'process': process,
            'config_path': config_path,
            'output_file': output_file,
            'start_time': datetime.now(),
            'status': 'running'
        }
        
        print(f"  → PID: {process.pid}, Output: {output_file}")
        
    def monitor_jobs(self, check_interval: int = 30) -> None:
        """Monitor job progress and report status."""
        print(f"\n🔍 Monitoring jobs every {check_interval} seconds...")
        print("Press Ctrl+C to stop monitoring and get final status")
        print("-" * 60)
        
        try:
            while self._any_jobs_running():
                self._check_job_status()
                self._print_status_summary()
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n⚠️  Monitoring interrupted by user")
            
        finally:
            self._final_status_report()
            
    def _any_jobs_running(self) -> bool:
        """Check if any jobs are still running."""
        return any(job['status'] == 'running' for job in self.jobs.values())
        
    def _check_job_status(self) -> None:
        """Check the status of all jobs."""
        for job_id, job in self.jobs.items():
            if job['status'] == 'running':
                try:
                    # Check if process is still alive
                    process = psutil.Process(job['pid'])
                    if not process.is_running():
                        job['status'] = 'completed'
                        job['end_time'] = datetime.now()
                        job['return_code'] = process.returncode if hasattr(process, 'returncode') else 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    job['status'] = 'completed'
                    job['end_time'] = datetime.now()
                    job['return_code'] = job['process'].poll()
                    
    def _print_status_summary(self) -> None:
        """Print a summary of job statuses."""
        running_jobs = [jid for jid, job in self.jobs.items() if job['status'] == 'running']
        completed_jobs = [jid for jid, job in self.jobs.items() if job['status'] == 'completed']
        
        print(f"\n📊 Status Update - {datetime.now().strftime('%H:%M:%S')}")
        print(f"Running jobs: {len(running_jobs)} {running_jobs}")
        print(f"Completed jobs: {len(completed_jobs)} {completed_jobs}")
        
        # Show resource usage for running jobs
        total_cpu = 0
        total_memory = 0
        
        for job_id in running_jobs:
            try:
                process = psutil.Process(self.jobs[job_id]['pid'])
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / 1024 / 1024
                total_cpu += cpu_percent
                total_memory += memory_mb
                print(f"  Job {job_id}: CPU {cpu_percent:.1f}%, Memory {memory_mb:.0f}MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        if running_jobs:
            print(f"Total resource usage: CPU {total_cpu:.1f}%, Memory {total_memory:.0f}MB")
            
    def _final_status_report(self) -> None:
        """Generate final status report and collect results."""
        print("\n" + "="*60)
        print("📋 FINAL STATUS REPORT")
        print("="*60)
        
        total_runtime = datetime.now() - self.start_time
        
        for job_id, job in sorted(self.jobs.items()):
            status_emoji = "✅" if job['status'] == 'completed' else "❌"
            runtime = (job.get('end_time', datetime.now()) - job['start_time'])
            
            print(f"\n{status_emoji} Job {job_id}:")
            print(f"  Config: {Path(job['config_path']).name}")
            print(f"  Runtime: {runtime}")
            print(f"  Output: {job['output_file']}")
            print(f"  PID: {job['pid']}")
            
            if job['status'] == 'completed':
                return_code = job.get('return_code', 'unknown')
                print(f"  Return code: {return_code}")
                
                # Show last few lines of output
                self._show_output_tail(job['output_file'])
            else:
                print(f"  Status: Still running")
                
        print(f"\n⏱️  Total experiment runtime: {total_runtime}")
        print(f"📁 All output files saved in: {self.output_dir.absolute()}")
        
        # Save summary to JSON
        self._save_results_summary(total_runtime)
        
    def _show_output_tail(self, output_file: Path, lines: int = 5) -> None:
        """Show the last few lines of an output file."""
        try:
            with open(output_file, 'r') as f:
                file_lines = f.readlines()
                if file_lines:
                    tail_lines = file_lines[-lines:]
                    print(f"  Last {len(tail_lines)} lines:")
                    for line in tail_lines:
                        print(f"    {line.rstrip()}")
                else:
                    print(f"  Output file is empty")
        except Exception as e:
            print(f"  Could not read output file: {e}")
            
    def _save_results_summary(self, total_runtime) -> None:
        """Save results summary to JSON file."""
        summary = {
            'experiment_info': {
                'num_jobs': self.num_jobs,
                'start_time': self.start_time.isoformat(),
                'total_runtime_seconds': total_runtime.total_seconds(),
                'config_base_path': self.config_base_path
            },
            'jobs': {}
        }
        
        for job_id, job in self.jobs.items():
            job_runtime = (job.get('end_time', datetime.now()) - job['start_time'])
            summary['jobs'][job_id] = {
                'config_path': job['config_path'],
                'output_file': str(job['output_file']),
                'pid': job['pid'],
                'status': job['status'],
                'start_time': job['start_time'].isoformat(),
                'runtime_seconds': job_runtime.total_seconds(),
                'return_code': job.get('return_code')
            }
            
        summary_file = self.output_dir / f"parallel_inference_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📄 Results summary saved to: {summary_file}")
        
    def cleanup_jobs(self) -> None:
        """Clean up any remaining jobs."""
        print("\n🧹 Cleaning up jobs...")
        
        for job_id, job in self.jobs.items():
            if job['status'] == 'running':
                try:
                    # Kill the process group to ensure child processes are also killed
                    os.killpg(os.getpgid(job['pid']), signal.SIGTERM)
                    print(f"  Terminated job {job_id} (PID: {job['pid']})")
                    time.sleep(1)
                    
                    # Force kill if still running
                    try:
                        os.killpg(os.getpgid(job['pid']), signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # Already dead
                        
                except (ProcessLookupError, PermissionError):
                    print(f"  Job {job_id} already terminated")
                    

def main():
    """Main function to parse arguments and run the parallel inference."""
    parser = argparse.ArgumentParser(
        description="Run parallel GPU inference jobs for ATLAS muon filtering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_parallel_inference.py -n 3
    Run 3 parallel jobs (configs 0, 1, 2)
    
  python run_parallel_inference.py -n 5 --output-dir results
    Run 5 jobs and save outputs to 'results' directory
    
  python run_parallel_inference.py -n 10 --no-monitor
    Run all 10 jobs without interactive monitoring
        """
    )
    
    parser.add_argument(
        '-n', '--num-jobs',
        type=int,
        required=True,
        help='Number of parallel jobs to run (1-10)'
    )
    
    parser.add_argument(
        '--config-base-path',
        type=str,
        default='/home/iwsatlas1/jrenusch/master_thesis/tracking/hepattn_muon/src/hepattn/experiments/atlas_muon/configs/parallelization_testing/atlas_muon_filtering_small_inference',
        help='Base path for config files (without number and .yaml extension)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save output files (default: current directory)'
    )
    
    parser.add_argument(
        '--check-interval',
        type=int,
        default=30,
        help='Interval in seconds between status checks (default: 30)'
    )
    
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Submit jobs and exit without monitoring'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.num_jobs < 1 or args.num_jobs > 10:
        print("❌ Error: Number of jobs must be between 1 and 10")
        return 1
        
    # Check if config files exist
    for i in range(args.num_jobs):
        config_path = f"{args.config_base_path}{i}.yaml"
        if not Path(config_path).exists():
            print(f"❌ Error: Config file not found: {config_path}")
            return 1
            
    try:
        # Create job manager and submit jobs
        job_manager = JobManager(
            num_jobs=args.num_jobs,
            config_base_path=args.config_base_path,
            output_dir=args.output_dir
        )
        
        job_manager.submit_jobs()
        
        if not args.no_monitor:
            job_manager.monitor_jobs(check_interval=args.check_interval)
        else:
            print(f"\n✅ {args.num_jobs} jobs submitted. Check output files for results.")
            print(f"📁 Output directory: {Path(args.output_dir).absolute()}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Script interrupted by user")
        if 'job_manager' in locals():
            job_manager.cleanup_jobs()
        return 130
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())
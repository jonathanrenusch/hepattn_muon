#!/usr/bin/env python3
"""
Test script to simulate corrupt trials for testing the cleanup script.
This adds some dummy trials after trial 61 for testing purposes.
"""

import sqlite3
from datetime import datetime


def add_test_trials(db_path: str, start_number: int = 62, count: int = 5):
    """Add some test trials for cleanup testing."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Adding {count} test trials starting from number {start_number}")
    
    try:
        for i in range(count):
            trial_number = start_number + i
            now = datetime.now().isoformat()
            
            # Insert trial
            cursor.execute("""
                INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete)
                VALUES (?, 1, 'COMPLETE', ?, ?)
            """, (trial_number, now, now))
            
            trial_id = cursor.lastrowid
            print(f"  Added trial {trial_number} (ID: {trial_id})")
            
            # Add some dummy parameters
            cursor.execute("""
                INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json)
                VALUES (?, 'test_param', 1.0, '{"name": "TestDistribution"}')
            """, (trial_id,))
            
            # Add dummy value
            cursor.execute("""
                INSERT INTO trial_values (trial_id, objective, value, value_type)
                VALUES (?, 0, 999.0, 'FINITE')
            """, (trial_id,))
        
        conn.commit()
        print(f"✅ Successfully added {count} test trials")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error adding test trials: {e}")
        raise
    
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add test trials for cleanup testing")
    parser.add_argument("--db-path", type=str, default="./optuna_study.db",
                       help="Path to the Optuna SQLite database")
    parser.add_argument("--start", type=int, default=62,
                       help="Starting trial number")
    parser.add_argument("--count", type=int, default=5,
                       help="Number of test trials to add")
    
    args = parser.parse_args()
    
    add_test_trials(args.db_path, args.start, args.count)
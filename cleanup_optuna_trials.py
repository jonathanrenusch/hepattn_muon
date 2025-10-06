#!/usr/bin/env python3
"""
Script to clean up corrupt Optuna trials from the database.
Removes all trials after a specified trial number.

Usage:
    python cleanup_optuna_trials.py --after 61 --db-path ./optuna_study.db
    python cleanup_optuna_trials.py --after 61 --db-path ./optuna_study.db --dry-run
"""

import argparse
import sqlite3
from typing import List, Tuple


def get_trials_to_delete(db_path: str, after_trial: int) -> List[Tuple[int, int]]:
    """Get list of trials to delete (trial_id, number pairs)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get trials after the specified number
    cursor.execute("SELECT trial_id, number FROM trials WHERE number > ? ORDER BY number", (after_trial,))
    trials_to_delete = cursor.fetchall()
    
    conn.close()
    return trials_to_delete


def delete_trials_after(db_path: str, after_trial: int, dry_run: bool = True) -> None:
    """Delete all trials after the specified trial number."""
    
    # First, check what we would delete
    trials_to_delete = get_trials_to_delete(db_path, after_trial)
    
    if not trials_to_delete:
        print(f"No trials found after trial number {after_trial}")
        return
    
    print(f"Found {len(trials_to_delete)} trials to delete:")
    for trial_id, number in trials_to_delete:
        print(f"  Trial ID: {trial_id}, Number: {number}")
    
    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***")
        print(f"Would delete {len(trials_to_delete)} trials and all their associated data")
        return
    
    # Confirm deletion
    response = input(f"\nAre you sure you want to delete {len(trials_to_delete)} trials? (yes/no): ")
    if response.lower() != 'yes':
        print("Deletion cancelled.")
        return
    
    # Perform deletion
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        total_deleted = 0
        
        for trial_id, number in trials_to_delete:
            print(f"Deleting trial {number} (ID: {trial_id})...")
            
            # Delete from all related tables in the correct order
            # (respecting foreign key constraints)
            
            # 1. Delete trial intermediate values
            cursor.execute("DELETE FROM trial_intermediate_values WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} intermediate values")
            
            # 2. Delete trial heartbeats
            cursor.execute("DELETE FROM trial_heartbeats WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} heartbeats")
            
            # 3. Delete trial values
            cursor.execute("DELETE FROM trial_values WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} trial values")
            
            # 4. Delete trial parameters
            cursor.execute("DELETE FROM trial_params WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} parameters")
            
            # 5. Delete trial system attributes
            cursor.execute("DELETE FROM trial_system_attributes WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} system attributes")
            
            # 6. Delete trial user attributes
            cursor.execute("DELETE FROM trial_user_attributes WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted {deleted} user attributes")
            
            # 7. Finally, delete the trial itself
            cursor.execute("DELETE FROM trials WHERE trial_id = ?", (trial_id,))
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"  - Deleted trial record")
                total_deleted += 1
        
        # Commit transaction
        conn.commit()
        print(f"\n✅ Successfully deleted {total_deleted} trials and all associated data")
        
    except Exception as e:
        # Rollback on error
        conn.rollback()
        print(f"\n❌ Error during deletion: {e}")
        print("All changes have been rolled back.")
        raise
    
    finally:
        conn.close()


def show_database_summary(db_path: str) -> None:
    """Show a summary of the database state."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get trial count and range
    cursor.execute("SELECT COUNT(*), MIN(number), MAX(number) FROM trials")
    count, min_num, max_num = cursor.fetchone()
    
    print(f"\nDatabase summary:")
    print(f"  Total trials: {count}")
    if count > 0:
        print(f"  Trial number range: {min_num} - {max_num}")
    
    # Show trials by state
    cursor.execute("SELECT state, COUNT(*) FROM trials GROUP BY state")
    states = cursor.fetchall()
    if states:
        print("  Trials by state:")
        for state, state_count in states:
            print(f"    {state}: {state_count}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Clean up corrupt Optuna trials")
    parser.add_argument("--after", type=int, required=True, 
                       help="Delete all trials after this trial number")
    parser.add_argument("--db-path", type=str, default="./optuna_study.db",
                       help="Path to the Optuna SQLite database")
    parser.add_argument("--dry-run", action="store_true", default=False,
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--summary", action="store_true", default=False,
                       help="Show database summary before and after")
    
    args = parser.parse_args()
    
    print(f"Optuna Database Cleanup Tool")
    print(f"Database: {args.db_path}")
    print(f"Target: Delete all trials after trial number {args.after}")
    
    if args.summary:
        print("\n" + "="*50)
        print("BEFORE CLEANUP:")
        show_database_summary(args.db_path)
    
    print("\n" + "="*50)
    delete_trials_after(args.db_path, args.after, args.dry_run)
    
    if args.summary and not args.dry_run:
        print("\n" + "="*50)
        print("AFTER CLEANUP:")
        show_database_summary(args.db_path)


if __name__ == "__main__":
    main()
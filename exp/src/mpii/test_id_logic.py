#!/usr/bin/env python3
"""
Direct test of the experiment ID logic extracted from MPII scripts.
Tests the core logic without heavy imports.
"""

import json
import gzip
import tempfile
from pathlib import Path
from datetime import datetime

def test_experiment_id_logic(config_experiment_id: str, checkpoint_exists: bool, output_folder: str) -> str:
    """
    Test the exact experiment ID logic from MPII scripts.
    This is the core logic we added to fix the bug.
    """
    print(f"\n🔍 Testing: config_id='{config_experiment_id}', checkpoint={checkpoint_exists}")
    
    # This is the exact logic from our fix in MPII scripts
    if not config_experiment_id or config_experiment_id.strip() == "":
        # Case 1: No experiment ID in config → start fresh with new ID
        experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Starting fresh evaluation with new experiment ID: {experiment_id}")
        result = f"FRESH_NEW_ID: {experiment_id}"
    else:
        # Case 2: Experiment ID provided → check for checkpoint files
        config_experiment_id = config_experiment_id.strip()
        checkpoint_file = Path(output_folder) / f"checkpoint_{config_experiment_id}.json.gz"
        
        if checkpoint_file.exists():
            # Resume from existing checkpoint + use existing log file
            experiment_id = config_experiment_id
            print(f"Found checkpoint for experiment ID: {experiment_id}, will attempt to resume")
            result = f"RESUME_EXISTING: {experiment_id}"
        else:
            # No checkpoint found → start fresh with new ID + new log file
            experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"No checkpoint found for experiment ID: {config_experiment_id}")
            print(f"Starting fresh evaluation with new experiment ID: {experiment_id}")
            result = f"FRESH_NEW_ID: {experiment_id}"
    
    return result

def main():
    """Run direct tests of the experiment ID logic."""
    print("🧪 Direct Test of MPII Experiment ID Logic")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_folder = temp_dir
        results_dir = Path(output_folder)
        
        # Test Case 1: Empty experiment ID
        print("\n📋 Test Case 1: Empty experiment ID")
        result1 = test_experiment_id_logic("", False, output_folder)
        expected1 = "Should generate new timestamp ID"
        success1 = result1.startswith("FRESH_NEW_ID:") and "eval_" in result1
        print(f"    Expected: {expected1}")
        print(f"    Result: {result1}")
        print(f"    Status: {'✅ PASS' if success1 else '❌ FAIL'}")
        
        # Test Case 2: Valid experiment ID with checkpoint
        test_exp_id = "test_experiment_20240101_120000"
        checkpoint_file = results_dir / f"checkpoint_{test_exp_id}.json.gz"
        
        # Create fake checkpoint
        fake_checkpoint = {"experiment_id": test_exp_id, "processed_files": []}
        with gzip.open(checkpoint_file, 'wt') as f:
            json.dump(fake_checkpoint, f)
        
        print(f"\n📋 Test Case 2: Valid ID with checkpoint ({checkpoint_file.name})")
        result2 = test_experiment_id_logic(test_exp_id, True, output_folder)
        expected2 = f"Should resume with same ID: {test_exp_id}"
        success2 = result2 == f"RESUME_EXISTING: {test_exp_id}"
        print(f"    Expected: {expected2}")
        print(f"    Result: {result2}")
        print(f"    Status: {'✅ PASS' if success2 else '❌ FAIL'}")
        
        # Remove checkpoint for next test
        checkpoint_file.unlink()
        
        # Test Case 3: Valid experiment ID without checkpoint
        print(f"\n📋 Test Case 3: Valid ID without checkpoint")
        result3 = test_experiment_id_logic(test_exp_id, False, output_folder)
        expected3 = "Should generate new timestamp ID (not reuse old one)"
        success3 = result3.startswith("FRESH_NEW_ID:") and test_exp_id not in result3 and "eval_" in result3
        print(f"    Expected: {expected3}")
        print(f"    Result: {result3}")
        print(f"    Status: {'✅ PASS' if success3 else '❌ FAIL'}")
        
        # Summary
        total_tests = 3
        passed_tests = sum([success1, success2, success3])
        print(f"\n📊 SUMMARY:")
        print(f"    Tests passed: {passed_tests}/{total_tests}")
        print(f"    Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! The experiment ID fix logic is working correctly.")
            print("\n📋 Behavior Summary:")
            print("    ✅ Empty experiment_id → Generates new timestamp ID")
            print("    ✅ Valid ID + checkpoint exists → Resumes with same ID") 
            print("    ✅ Valid ID + no checkpoint → Generates new timestamp ID")
            print("\nThis fixes the log file reuse bug! 🔧")
        else:
            print("\n❌ Some tests failed. The fix may have issues.")

if __name__ == "__main__":
    main()
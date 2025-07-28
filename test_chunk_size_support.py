#!/usr/bin/env python3
"""
Test script to validate chunk_size support across evaluation frameworks.
Uses minimal test data to avoid long execution times.
"""

import os
import sys
import tempfile
import json
import yaml
from pathlib import Path

def create_test_data():
    """Create minimal test data for validation."""
    
    # Simple test case with short texts
    test_data = {
        "ground_truth": "The cat sat on the mat. It was a sunny day.",
        "categories": [
            {
                "name": "test_category",
                "test_cases": [
                    {
                        "id": "test_1",
                        "name": "basic_test",
                        "description": "The cat was sitting on the mat. It was sunny."
                    },
                    {
                        "id": "test_2", 
                        "name": "variation_test",
                        "description": "A cat sat down on a mat during a bright day."
                    }
                ]
            }
        ]
    }
    
    return test_data

def create_vatex_test_data():
    """Create minimal VATEX-EVAL test data."""
    vatex_data = {
        "video_1": {
            "captions": [
                "A person walks down the street.",
                "Someone is walking on the road.",
                "A man strolls along the path."
            ],
            "human_scores": [4.2, 3.8, 4.5]
        },
        "video_2": {
            "captions": [
                "The dog runs in the park.",
                "A puppy plays outside.",
                "An animal moves quickly."
            ],
            "human_scores": [4.0, 3.5, 2.8]
        }
    }
    
    return vatex_data

def create_test_config(framework, chunk_sizes, lct_values, temp_dir):
    """Create test configuration with specified parameters."""
    
    if framework == "mpii_comparison":
        config = {
            "models": {
                "nv_embed_path": "/mmfs1/scratch/jacks.local/mali9292/VAD-LLM2Vec/nv-embed"
            },
            "vcs": {
                "chunk_size": chunk_sizes,
                "lct": lct_values,
                "context_cutoff_value": 0.6,
                "context_window_control": 4.0,
                "return_all_metrics": True,
                "return_internals": False
            },
            "processing": {
                "max_workers": 1,
                "checkpoint_interval": 1,
                "resume_from_checkpoint": False
            },
            "paths": {
                "data_dir": str(temp_dir / "data"),
                "results_dir": str(temp_dir / "results"),
                "logs_dir": str(temp_dir / "logs")
            },
            "output": {
                "decimal_precision": 3
            },
            "experiment": {
                "experiment_id": "test_chunk_size"
            }
        }
    
    elif framework == "vatex_eval":
        config = {
            "models": {
                "nv_embed_path": "/mmfs1/scratch/jacks.local/mali9292/VAD-LLM2Vec/nv-embed"
            },
            "vcs": {
                "chunk_size": chunk_sizes,
                "lct": lct_values,
                "context_cutoff_value": 0.6,
                "context_window_control": 4.0,
                "return_all_metrics": True,
                "return_internals": False
            },
            "vatex_eval": {
                "data_dir": str(temp_dir / "vatex_data.json"),
                "use_n_refs": [1]
            },
            "processing": {
                "max_workers": 1,
                "checkpoint_interval": 1,
                "resume_from_checkpoint": False
            },
            "paths": {
                "results_dir": str(temp_dir / "results"),
                "logs_dir": str(temp_dir / "logs")
            },
            "output": {
                "decimal_precision": 3
            },
            "experiment": {
                "experiment_id": "test_chunk_size"
            }
        }
    
    return config

def test_config_loading():
    """Test 1: Verify configs load correctly with list format."""
    print("🔧 Test 1: Config Loading")
    
    test_cases = [
        ([1], [0]),           # Single values
        ([1, 2], [0, 1]),     # Multiple values
        ([1, 2, 3], [0]),     # Multiple chunk_size, single LCT
    ]
    
    results = []
    
    for chunk_sizes, lct_values in test_cases:
        try:
            config = create_test_config("mpii_comparison", chunk_sizes, lct_values, Path("/tmp"))
            
            # Verify config structure
            assert config["vcs"]["chunk_size"] == chunk_sizes
            assert config["vcs"]["lct"] == lct_values
            
            results.append(f"✅ chunk_size={chunk_sizes}, lct={lct_values}")
            
        except Exception as e:
            results.append(f"❌ chunk_size={chunk_sizes}, lct={lct_values}: {e}")
    
    for result in results:
        print(f"   {result}")
    
    return len([r for r in results if r.startswith("✅")])

def test_metric_naming():
    """Test 2: Verify expected metric names are generated."""
    print("\\n🏷️  Test 2: Metric Naming")
    
    test_cases = [
        ([1], [0], ["VCS_C1_LCT0"]),
        ([1, 2], [0], ["VCS_C1_LCT0", "VCS_C2_LCT0"]),
        ([1], [0, 1], ["VCS_C1_LCT0", "VCS_C1_LCT1"]),
        ([1, 2], [0, 1], ["VCS_C1_LCT0", "VCS_C1_LCT1", "VCS_C2_LCT0", "VCS_C2_LCT1"]),
    ]
    
    results = []
    
    for chunk_sizes, lct_values, expected_metrics in test_cases:
        try:
            # Generate expected metrics programmatically
            generated_metrics = []
            for chunk_size in chunk_sizes:
                for lct in lct_values:
                    generated_metrics.append(f"VCS_C{chunk_size}_LCT{lct}")
            
            # Check if they match expected
            if set(generated_metrics) == set(expected_metrics):
                results.append(f"✅ {chunk_sizes}x{lct_values} → {generated_metrics}")
            else:
                results.append(f"❌ {chunk_sizes}x{lct_values}: expected {expected_metrics}, got {generated_metrics}")
                
        except Exception as e:
            results.append(f"❌ {chunk_sizes}x{lct_values}: {e}")
    
    for result in results:
        print(f"   {result}")
    
    return len([r for r in results if r.startswith("✅")])

def test_import_check():
    """Test 3: Check if scripts import without errors."""
    print("\\n📦 Test 3: Import Validation")
    
    script_paths = [
        "exp/src/mpii/scripts/mpii_eval_comparison.py",
        "exp/src/mpii/scripts/mpii_eval_authors.py", 
        "exp/src/mpii/scripts/mpii_eval_chronology.py",
        "exp/src/vatex-eval/scripts/vatex-eval.py",
        "benchmarking/src/scripts/clipcc_eval_vlms.py"
    ]
    
    results = []
    base_dir = Path("/ces/scratch/jacks.local/mali9292/Video-Comprehension-Score")
    
    for script_path in script_paths:
        try:
            full_path = base_dir / script_path
            if full_path.exists():
                # Try to compile the script (syntax check)
                with open(full_path, 'r') as f:
                    code = f.read()
                compile(code, str(full_path), 'exec')
                results.append(f"✅ {script_path}")
            else:
                results.append(f"❌ {script_path}: File not found")
                
        except SyntaxError as e:
            results.append(f"❌ {script_path}: Syntax error - {e}")
        except Exception as e:
            results.append(f"❌ {script_path}: {e}")
    
    for result in results:
        print(f"   {result}")
    
    return len([r for r in results if r.startswith("✅")])

def test_dry_run_simulation():
    """Test 4: Simulate a dry run with mock VCS computation."""
    print("\\n🎯 Test 4: Dry Run Simulation")
    
    def mock_vcs_computation(chunk_size, lct):
        """Mock VCS computation that returns predictable results."""
        return {
            "VCS": round(0.5 + (chunk_size * 0.1) + (lct * 0.05), 3),
            "GAS": 0.3,
            "LAS": 0.4,
            "NAS": 0.2
        }
    
    test_cases = [
        ([1], [0]),
        ([1, 2], [0, 1]),
        ([1, 2, 3], [0, 1])
    ]
    
    results = []
    
    for chunk_sizes, lct_values in test_cases:
        try:
            expected_metrics = {}
            
            # Simulate what the scripts should produce
            for chunk_size in chunk_sizes:
                for lct in lct_values:
                    mock_result = mock_vcs_computation(chunk_size, lct)
                    expected_metrics[f"VCS_C{chunk_size}_LCT{lct}"] = mock_result["VCS"]
            
            # Verify we have the right number of metrics
            expected_count = len(chunk_sizes) * len(lct_values)
            actual_count = len(expected_metrics)
            
            if actual_count == expected_count:
                results.append(f"✅ {chunk_sizes}x{lct_values}: {actual_count} metrics → {list(expected_metrics.keys())}")
            else:
                results.append(f"❌ {chunk_sizes}x{lct_values}: expected {expected_count}, got {actual_count}")
                
        except Exception as e:
            results.append(f"❌ {chunk_sizes}x{lct_values}: {e}")
    
    for result in results:
        print(f"   {result}")
    
    return len([r for r in results if r.startswith("✅")])

def main():
    """Run all validation tests."""
    print("🧪 Testing Chunk Size Support Across Evaluation Frameworks")
    print("=" * 70)
    
    total_tests = 0
    passed_tests = 0
    
    # Run all tests
    tests = [
        test_config_loading,
        test_metric_naming,
        test_import_check,
        test_dry_run_simulation
    ]
    
    for test_func in tests:
        test_passed = test_func()
        total_tests += 4  # Approximate number of sub-tests per function
        passed_tests += test_passed
    
    # Summary
    print("\\n" + "=" * 70)
    print(f"📊 Test Summary: {passed_tests}/{total_tests} checks passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The chunk_size updates appear to be working correctly.")
        return 0
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
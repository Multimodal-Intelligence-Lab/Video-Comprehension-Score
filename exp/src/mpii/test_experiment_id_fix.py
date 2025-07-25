#!/usr/bin/env python3
"""
Comprehensive test script for MPII experiment ID handling fix.

Tests all scenarios:
1. Null/empty experiment_id in config
2. Valid experiment_id with existing checkpoint 
3. Valid experiment_id without checkpoint

Tests all MPII scripts:
- mpii_eval_comparison.py
- mpii_eval_authors.py  
- mpii_eval_addition.py
- mpii_eval_deletion.py
- mpii_eval_chronology.py
"""

import os
import sys
import yaml
import tempfile
import shutil
import json
import gzip
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

class MPIIExperimentIDTester:
    """Test the experiment ID handling fix across all MPII scripts."""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.test_temp_dir = Path(tempfile.mkdtemp(prefix="mpii_test_"))
        self.test_results = []
        
        # MPII scripts to test
        self.scripts = {
            'comparison': 'scripts/mpii_eval_comparison.py',
            'authors': 'scripts/mpii_eval_authors.py', 
            'addition': 'scripts/mpii_eval_addition.py',
            'deletion': 'scripts/mpii_eval_deletion.py',
            'chronology': 'scripts/mpii_eval_chronology.py'
        }
        
        print(f"🧪 MPII Experiment ID Test Suite")
        print(f"📁 Test directory: {self.test_temp_dir}")
        print("=" * 80)
    
    def create_minimal_test_config(self, experiment_id: str = None) -> Dict:
        """Create minimal config for testing (no actual model paths)."""
        config = {
            'experiment': {
                'experiment_id': experiment_id if experiment_id else ""
            },
            'paths': {
                'data_dir': str(self.test_temp_dir / "data"),
                'results_dir': str(self.test_temp_dir / "results"),
                'logs_dir': str(self.test_temp_dir / "logs")
            },
            'models': {
                'nv_embed_path': "/fake/path/nv-embed",  # Won't be used in init test
                'sat_model': "sat-12l-sm"
            },
            'vcs': {
                'chunk_size': 1,
                'lct_values': [0, 1],
                'context_cutoff_value': 0.6,
                'context_window_control': 4.0
            },
            'processing': {
                'max_workers': 2,
                'resume_from_checkpoint': True
            },
            'output': {
                'decimal_precision': 3
            },
            'embedding': {
                'batch_size': 2,
                'max_length': 1024
            }
        }
        return config
    
    def create_fake_checkpoint(self, results_dir: Path, experiment_id: str) -> None:
        """Create a fake checkpoint file to test resumption logic."""
        results_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "experiment_id": experiment_id,
            "processed_files": ["file1.json", "file2.json"],
            "failed_files": [],
            "processing_stats": {
                "files_completed": 2,
                "files_failed": 0
            }
        }
        
        checkpoint_file = results_dir / f"checkpoint_{experiment_id}.json.gz"
        with gzip.open(checkpoint_file, 'wt') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"    📝 Created fake checkpoint: {checkpoint_file}")
    
    def test_script_initialization(self, script_name: str, config: Dict, test_case: str) -> Tuple[bool, str, str]:
        """Test script initialization with given config. Returns (success, experiment_id, output)."""
        try:
            config_file = self.test_temp_dir / f"test_config_{script_name}_{test_case}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Create minimal data structure to avoid import errors
            data_dir = Path(config['paths']['data_dir'])
            data_dir.mkdir(parents=True, exist_ok=True)
            (data_dir / "dummy.json").write_text('{"test": "data"}')
            
            # Import the script class dynamically and test initialization
            script_path = self.script_dir / self.scripts[script_name]
            if not script_path.exists():
                return False, "MISSING", f"Script not found: {script_path}"
            
            # Create a test script to capture the experiment ID
            test_script_content = f'''
import sys
import os
from pathlib import Path
import yaml

# Add the mpii src directory to path  
mpii_src = Path(__file__).parent
sys.path.insert(0, str(mpii_src))
sys.path.insert(0, str(mpii_src / "utils"))

try:
    # Import the main EvaluationPipeline class from each script
    module_name = f"mpii_eval_{script_name}"
    script_path = mpii_src / "scripts" / f"{module_name}.py"
    
    spec = __import__("importlib.util").util.spec_from_file_location(module_name, script_path)
    module = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    # Load config
    with open("{config_file}", 'r') as f:
        config = yaml.safe_load(f)
    
    # Try to initialize (just to test experiment ID logic)
    evaluator = module.EvaluationPipeline(config)
    print(f"EXPERIMENT_ID:{evaluator.experiment_id}")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''
            
            test_script_file = self.test_temp_dir / f"test_{script_name}_{test_case}.py"
            with open(test_script_file, 'w') as f:
                f.write(test_script_content)
            
            # Run the test script and capture output
            result = subprocess.run(
                [sys.executable, str(test_script_file)],
                cwd=str(self.script_dir),
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Extract experiment ID from output
            experiment_id = "UNKNOWN"
            for line in output.split('\\n'):
                if line.startswith("EXPERIMENT_ID:"):
                    experiment_id = line.replace("EXPERIMENT_ID:", "")
                    break
            
            success = result.returncode == 0 and "ERROR:" not in output and experiment_id != "UNKNOWN"
            
            return success, experiment_id, output
            
        except Exception as e:
            return False, "ERROR", str(e)
    
    def run_test_case(self, test_name: str, experiment_id: str, create_checkpoint: bool) -> None:
        """Run a single test case across all scripts."""
        print(f"\\n🔍 Test Case: {test_name}")
        print(f"   Config experiment_id: '{experiment_id}'")
        print(f"   Checkpoint exists: {create_checkpoint}")
        print("-" * 60)
        
        results = {}
        
        for script_name in self.scripts.keys():
            print(f"\\n  📋 Testing {script_name}...")
            
            # Create config
            config = self.create_minimal_test_config(experiment_id)
            results_dir = Path(config['paths']['results_dir'])
            
            # Create checkpoint if requested
            if create_checkpoint and experiment_id:
                self.create_fake_checkpoint(results_dir, experiment_id)
            
            # Test script initialization
            success, actual_experiment_id, output = self.test_script_initialization(script_name, config, test_name)
            
            # Analyze result
            if success:
                # Check if behavior matches expectations
                if test_name == "null_experiment_id":
                    expected = "Generated new ID (timestamp format)"
                    correct = actual_experiment_id.startswith("eval_") and len(actual_experiment_id) == 19
                elif test_name == "valid_id_with_checkpoint":
                    expected = f"Use provided ID: {experiment_id}"
                    correct = actual_experiment_id == experiment_id
                elif test_name == "valid_id_without_checkpoint":
                    expected = "Generated new ID (timestamp format)"
                    correct = actual_experiment_id.startswith("eval_") and actual_experiment_id != experiment_id
                else:
                    expected = "Unknown test case"
                    correct = False
                
                status = "✅ PASS" if correct else "❌ FAIL"
                print(f"    {status} Expected: {expected}")
                print(f"    📋 Actual ID: {actual_experiment_id}")
                
                if "Found checkpoint" in output:
                    print(f"    💾 Checkpoint detected: YES")
                elif "No checkpoint found" in output:
                    print(f"    💾 Checkpoint detected: NO")
                elif "Starting fresh" in output:
                    print(f"    💾 Starting fresh evaluation")
                
                results[script_name] = {
                    'success': True,
                    'correct_behavior': correct,
                    'experiment_id': actual_experiment_id,
                    'output': output[:200] + "..." if len(output) > 200 else output
                }
            else:
                print(f"    ❌ FAIL - Script initialization failed")
                print(f"    📋 Error: {output[:200]}...")
                results[script_name] = {
                    'success': False,
                    'correct_behavior': False,
                    'experiment_id': actual_experiment_id,
                    'output': output
                }
        
        self.test_results.append({
            'test_name': test_name,
            'results': results
        })
        
        # Clean up for next test
        if results_dir.exists():
            shutil.rmtree(results_dir, ignore_errors=True)
    
    def run_all_tests(self) -> None:
        """Run all test scenarios."""
        print("🚀 Starting comprehensive experiment ID tests...")
        
        # Test Case 1: Null/empty experiment_id
        self.run_test_case(
            test_name="null_experiment_id",
            experiment_id="",  # Empty string
            create_checkpoint=False
        )
        
        # Test Case 2: Valid experiment_id with existing checkpoint
        test_exp_id = "test_experiment_20240101_120000"
        self.run_test_case(
            test_name="valid_id_with_checkpoint", 
            experiment_id=test_exp_id,
            create_checkpoint=True
        )
        
        # Test Case 3: Valid experiment_id without checkpoint
        self.run_test_case(
            test_name="valid_id_without_checkpoint",
            experiment_id=test_exp_id,
            create_checkpoint=False
        )
    
    def print_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        total_tests = 0
        passed_tests = 0
        
        for test_result in self.test_results:
            print(f"\\n🔍 {test_result['test_name']}:")
            
            for script_name, result in test_result['results'].items():
                total_tests += 1
                status = "✅" if result['success'] and result['correct_behavior'] else "❌"
                print(f"  {status} {script_name}: {result['experiment_id']}")
                
                if result['success'] and result['correct_behavior']:
                    passed_tests += 1
                elif not result['success']:
                    print(f"    ⚠️  Initialization failed")
                elif not result['correct_behavior']:
                    print(f"    ⚠️  Incorrect behavior")
        
        print(f"\\n📈 Overall Results:")
        print(f"   Tests passed: {passed_tests}/{total_tests}")
        print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")
        
        if passed_tests == total_tests:
            print("\\n🎉 ALL TESTS PASSED! The experiment ID fix is working correctly.")
        else:
            print(f"\\n⚠️  {total_tests - passed_tests} tests failed. Please check the implementation.")
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self.test_temp_dir.exists():
            shutil.rmtree(self.test_temp_dir, ignore_errors=True)
            print(f"\\n🧹 Cleaned up test directory: {self.test_temp_dir}")

def main():
    """Run the comprehensive test suite."""
    tester = MPIIExperimentIDTester()
    
    try:
        tester.run_all_tests()
        tester.print_summary()
    except KeyboardInterrupt:
        print("\\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tester.cleanup()

if __name__ == "__main__":
    main()
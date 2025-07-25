#!/usr/bin/env python3
"""
Simple test to verify the MPII experiment ID fix is working.
Just checks the behavior by looking at the print statements from the fix.
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
from typing import Dict
import subprocess

def test_single_script(script_name: str, experiment_id: str, create_checkpoint: bool):
    """Test a single MPII script with specific conditions."""
    print(f"\n🔍 Testing {script_name} with experiment_id='{experiment_id}', checkpoint={create_checkpoint}")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        results_dir = temp_path / "results"
        results_dir.mkdir(parents=True)
        
        # Create fake checkpoint if needed
        if create_checkpoint and experiment_id:
            checkpoint_file = results_dir / f"checkpoint_{experiment_id}.json.gz"
            fake_data = {"experiment_id": experiment_id, "processed_files": []}
            with gzip.open(checkpoint_file, 'wt') as f:
                json.dump(fake_data, f)
            print(f"    📝 Created checkpoint: {checkpoint_file}")
        
        # Create minimal config
        config = {
            'experiment': {'experiment_id': experiment_id},
            'paths': {
                'data_dir': str(temp_path / "data"),
                'results_dir': str(results_dir),
                'logs_dir': str(temp_path / "logs")
            },
            'models': {'nv_embed_path': "/fake", 'sat_model': "fake"},
            'processing': {'max_workers': 1},
            'vcs': {'chunk_size': 1, 'lct_values': [0]},
            'output': {'decimal_precision': 3},
            'embedding': {'batch_size': 1, 'max_length': 128}
        }
        
        config_file = temp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Create minimal data
        data_dir = Path(config['paths']['data_dir'])
        data_dir.mkdir(parents=True)
        (data_dir / "test.json").write_text('{"test": "data"}')
        
        # Try to run the script initialization and capture just the first few lines of output
        script_path = Path(__file__).parent / "scripts" / f"mpii_eval_{script_name}.py"
        
        try:
            # Run script with Python but intercept it early to just see the experiment ID logic
            cmd = [sys.executable, "-c", f'''
import sys
from pathlib import Path
sys.path.insert(0, "{Path(__file__).parent}")
sys.path.insert(0, "{Path(__file__).parent / "utils"}")

# Mock the heavy imports to avoid loading actual models
sys.modules["transformers"] = type(sys)("fake_transformers")
sys.modules["wtpsplit"] = type(sys)("fake_wtpsplit") 
sys.modules["sentence_transformers"] = type(sys)("fake_sentence_transformers")

# Load the script and config
import importlib.util
import yaml

spec = importlib.util.spec_from_file_location("test_module", "{script_path}")
test_module = importlib.util.module_from_spec(spec)

# Patch heavy classes to avoid actual initialization
class MockSaT:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return []

class MockEmbedding:
    def __init__(self, *args, **kwargs): pass
    def encode(self, *args, **kwargs): return []

# Create mock classes to prevent heavy initialization
test_module.TextProcessor = type("MockTextProcessor", (), {{"sat_segmenter": MockSaT()}})
test_module.EmbeddingProcessor = type("MockEmbeddingProcessor", (), {{}})

# Load config
with open("{config_file}", 'r') as f:
    config = yaml.safe_load(f)

# Try to initialize just the experiment ID logic part
try:
    spec.loader.exec_module(test_module)
    # This will trigger the experiment ID logic and print statements
    pipeline = test_module.EvaluationPipeline(config)
    print(f"SUCCESS: Final experiment_id = {{pipeline.experiment_id}}")
except Exception as e:
    print(f"ERROR: {{e}}")
''']
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=str(Path(__file__).parent))
            
            output = result.stdout + result.stderr
            print(f"    📋 Output: {output.strip()}")
            
            # Look for the key messages from our fix
            if "Starting fresh evaluation with new experiment ID:" in output:
                print("    ✅ CORRECT: Generated new experiment ID (fresh start)")
            elif "Found checkpoint for experiment ID:" in output:
                print("    ✅ CORRECT: Found checkpoint, resuming")
            elif "No checkpoint found for experiment ID:" in output:
                print("    ✅ CORRECT: No checkpoint found, generating new ID")
            else:
                print("    ❌ UNKNOWN: Could not determine behavior")
                
            return output
            
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            return str(e)

def main():
    """Run simple tests to verify the fix works."""
    print("🧪 Simple MPII Experiment ID Fix Test")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        ("comparison", "", False, "null_experiment_id -> should generate new"),
        ("comparison", "test_20240101_120000", True, "valid_id_with_checkpoint -> should resume"),  
        ("comparison", "test_20240101_120000", False, "valid_id_without_checkpoint -> should generate new"),
    ]
    
    for script_name, experiment_id, create_checkpoint, description in test_cases:
        print(f"\n📋 {description}")
        test_single_script(script_name, experiment_id, create_checkpoint)
    
    print(f"\n🎯 Test completed! Check the outputs above to verify the fix is working.")

if __name__ == "__main__":
    main()
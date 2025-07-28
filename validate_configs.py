#!/usr/bin/env python3
"""
Quick validation script to test configuration loading across all frameworks.
"""

import yaml
from pathlib import Path

def validate_config_file(config_path):
    """Validate a single config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check VCS section
        if 'vcs' not in config:
            return False, "Missing 'vcs' section"
        
        vcs_config = config['vcs']
        
        # Check chunk_size format
        if 'chunk_size' not in vcs_config:
            return False, "Missing 'chunk_size' parameter"
        
        chunk_size = vcs_config['chunk_size']
        if not isinstance(chunk_size, list):
            return False, f"chunk_size should be a list, got {type(chunk_size)}"
        
        # Check lct format  
        lct_param = None
        if 'lct' in vcs_config:
            lct_param = 'lct'
        elif 'lct_values' in vcs_config:
            lct_param = 'lct_values'
        else:
            return False, "Missing LCT parameter (should be 'lct')"
        
        lct = vcs_config[lct_param]
        if not isinstance(lct, list):
            return False, f"{lct_param} should be a list, got {type(lct)}"
        
        # Calculate expected metrics
        expected_metrics = []
        for c in chunk_size:
            for l in lct:
                expected_metrics.append(f"VCS_C{c}_LCT{l}")
        
        return True, f"✅ chunk_size={chunk_size}, lct={lct} → {len(expected_metrics)} metrics: {expected_metrics}"
        
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Validate all configuration files."""
    print("🔍 Validating Configuration Files")
    print("=" * 50)
    
    # Find all config files
    project_root = Path("/ces/scratch/jacks.local/mali9292/Video-Comprehension-Score")
    
    config_files = [
        # VATEX-EVAL
        "exp/src/vatex-eval/config/vatex-eval.yaml",
        
        # MPII
        "exp/src/mpii/config/_base.yaml",
        "exp/src/mpii/config/addition.yaml", 
        "exp/src/mpii/config/deletion.yaml",
        
        # Benchmarking
        "benchmarking/src/config/clipcc_eval_vlms.yaml"
    ]
    
    results = []
    
    for config_file in config_files:
        config_path = project_root / config_file
        print(f"\\n📋 Checking: {config_file}")
        
        if not config_path.exists():
            print("   ❌ File not found")
            results.append(False)
            continue
        
        valid, message = validate_config_file(config_path)
        if valid:
            print(f"   {message}")
            results.append(True)
        else:
            print(f"   ❌ {message}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\\n" + "=" * 50)
    print(f"📊 Configuration Validation: {passed}/{total} files passed")
    
    if passed == total:
        print("🎉 All configuration files are valid!")
        print("\\n✅ Ready for testing with:")
        print("   • Consistent chunk_size parameter format (lists)")
        print("   • Consistent lct parameter naming")
        print("   • Expected VCS_C{chunk_size}_LCT{lct} metric generation")
    else:
        print(f"⚠️  {total - passed} configuration files need fixes")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    exit(main())
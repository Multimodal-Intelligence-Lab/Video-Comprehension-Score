"""
Core utilities for MPII Ablation Study Framework.

This module contains shared classes and utilities for ablation experiments,
providing consistent functionality across all ablation evaluation scripts.
"""

import torch
import torch.nn.functional as F
import re
import os
import time
import pickle
import yaml
import string
import contractions
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel
from wtpsplit import SaT
from typing import List, Dict, Optional, Any
from copy import deepcopy
import json
import gzip

# ============================================================================
# CONSTANTS
# ============================================================================

# Punctuation handling - exclude apostrophes for text processing
PUNCTUATIONS = set(string.punctuation) - {"'"}

# Output formatting
DECIMAL_PRECISION = 3

# Metrics ordering for consistent output
ABLATION_1_METRICS_ORDER = [
    "Precision NAS-D", "Recall NAS-D", "NAS-D",
    "Precision NAS-L", "Recall NAS-L", "NAS-L", "NAS",
    "Precision LAS", "Recall LAS", "LAS",
    "GAS", "GAS-LAS-Scaled", "VCS"
]

ABLATION_2_METRICS_ORDER = [
    "GAS", "LAS", "NAS-D", "NAS-L", "NAS",
    "NAS\n+LAS(S)", "GAS\n+LAS(S)", "GAS\n+NAS-L(S)", 
    "GAS\n+NAS-D(S)", "GAS\n+NAS(S)",
    "GAS\n+LAS(S)\n+NAS-D(S)", "GAS\n+LAS(S)\n+NAS-L(S)",
    "GAS\n+LAS(S)\n+(NAS-D\n+NAS-L)(S)"
]

# Default segmenter function
DEFAULT_SEGMENTER_FUNCTION = "sat"


def deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries. Override dict takes precedence.
    
    Args:
        base_dict: Base dictionary with default values
        override_dict: Override dictionary with specific values
        
    Returns:
        Merged dictionary with override values taking precedence
    """
    result = deepcopy(base_dict)
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge(result[key], value)
        else:
            # Override with new value
            result[key] = deepcopy(value)
    
    return result


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_path)
            
            # Load configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            ConfigLoader._validate_config(config)
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate configuration structure and required fields."""
        required_sections = ['paths', 'processing', 'vcs', 'output', 'models']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate paths
        paths = config['paths']
        required_paths = ['data_dir', 'results_dir']
        for path_key in required_paths:
            if path_key not in paths:
                raise ValueError(f"Missing required path: {path_key}")
        
        # Validate models
        models = config['models']
        required_models = ['nv_embed_path', 'sat_model']
        for model_key in required_models:
            if model_key not in models:
                raise ValueError(f"Missing required model: {model_key}")


class CheckpointManager:
    """Enhanced checkpoint manager optimized for large-scale processing (MPII-style).
    
    Features:
    - Memory-efficient storage (metadata only, not full results)
    - Thread-safe operations with locks
    - Integrity validation with checksums
    - Configuration consistency validation
    - Compressed storage format
    - Temporary JSON results directory
    """
    
    def __init__(self, checkpoint_dir: str, experiment_id: str, config_hash: str = None):
        import threading
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.config_hash = config_hash  # For configuration consistency validation
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{experiment_id}.json.gz"
        self.results_dir = self.checkpoint_dir / f"results_{experiment_id}"
        self.results_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._last_save_time = 0
        self._adaptive_interval = 5  # Start with 5, adapt based on processing speed
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for integrity validation."""
        import hashlib
        return hashlib.sha256(data).hexdigest()
    
    def _generate_config_hash(self, config: Dict) -> str:
        """Generate a hash of the configuration for consistency validation."""
        import hashlib
        import json
        
        # Create a normalized config string (sorted keys, exclude volatile fields)
        config_copy = config.copy()
        config_copy.pop('experiment', None)  # Remove experiment-specific fields
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_checkpoint(
        self, 
        processed_test_cases: List[str], 
        failed_test_cases: List[str] = None,
        current_ablation_type: str = None,
        processing_stats: Dict = None
    ) -> None:
        """Save lightweight checkpoint with metadata only.
        
        Args:
            processed_test_cases: List of successfully processed test case IDs
            failed_test_cases: List of test case IDs that failed processing (optional)
            current_ablation_type: Current ablation type being processed
            processing_stats: Processing statistics for adaptive intervals
        """
        import json
        import gzip
        import time
        
        with self._lock:
            # Adaptive checkpoint interval based on processing speed
            current_time = time.time()
            if processing_stats and self._last_save_time > 0:
                time_since_last = current_time - self._last_save_time
                avg_time_per_case = processing_stats.get('avg_time_per_case', 60)
                # Adjust interval: save every 2-3 minutes of processing time
                target_interval_time = 150  # 2.5 minutes
                self._adaptive_interval = max(3, int(target_interval_time / avg_time_per_case))
            
            self._last_save_time = current_time
            
            checkpoint_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "processed_test_cases": processed_test_cases,
                "failed_test_cases": failed_test_cases or [],
                "current_ablation_type": current_ablation_type,
                "processing_stats": processing_stats or {},
                "adaptive_interval": self._adaptive_interval
            }
            
            # Atomic write using temporary file with compression
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            try:
                json_data = json.dumps(checkpoint_data, indent=2)
                compressed_data = gzip.compress(json_data.encode('utf-8'))
                
                # Calculate checksum
                checksum = self._calculate_checksum(compressed_data)
                
                # Write checkpoint with checksum
                with open(temp_file, 'wb') as f:
                    f.write(compressed_data)
                
                # Write checksum file
                checksum_file = temp_file.with_suffix('.checksum')
                with open(checksum_file, 'w') as f:
                    f.write(checksum)
                
                # Atomic move
                temp_file.replace(self.checkpoint_file)
                checksum_file.replace(self.checkpoint_file.with_suffix('.json.gz.checksum'))
                
            except Exception as e:
                # Cleanup on failure
                for temp in [temp_file, temp_file.with_suffix('.checksum')]:
                    if temp.exists():
                        temp.unlink()
                raise e
    
    def load_checkpoint(self, config: Dict = None) -> Optional[Dict]:
        """Load and validate checkpoint.
        
        Args:
            config: Current configuration for consistency validation
            
        Returns:
            Dictionary with checkpoint state or None if invalid/missing
        """
        import json
        import gzip
        
        if not self.checkpoint_file.exists():
            return None
        
        checksum_file = self.checkpoint_file.with_suffix('.json.gz.checksum')
        
        try:
            with self._lock:
                # Read and validate checksum
                if checksum_file.exists():
                    with open(checksum_file, 'r') as f:
                        expected_checksum = f.read().strip()
                    
                    with open(self.checkpoint_file, 'rb') as f:
                        data = f.read()
                    
                    actual_checksum = self._calculate_checksum(data)
                    if actual_checksum != expected_checksum:
                        print("Warning: Checkpoint integrity check failed, starting fresh")
                        return None
                else:
                    # Old checkpoint without checksum - load anyway but warn
                    with open(self.checkpoint_file, 'rb') as f:
                        data = f.read()
                    print("Warning: Loading checkpoint without integrity validation")
                
                # Decompress and parse
                try:
                    json_data = gzip.decompress(data).decode('utf-8')
                except gzip.BadGzipFile:
                    # Handle old uncompressed checkpoints
                    json_data = data.decode('utf-8')
                
                checkpoint_data = json.loads(json_data)
                
                # Validate configuration consistency
                if config and self.config_hash:
                    current_config_hash = self._generate_config_hash(config)
                    if checkpoint_data.get('config_hash') != current_config_hash:
                        print("Warning: Configuration has changed since checkpoint, starting fresh")
                        return None
                
                # Update adaptive interval if available
                if 'adaptive_interval' in checkpoint_data:
                    self._adaptive_interval = checkpoint_data['adaptive_interval']
                
                return {
                    "processed_test_cases": checkpoint_data.get("processed_test_cases", []),
                    "failed_test_cases": checkpoint_data.get("failed_test_cases", []),
                    "current_ablation_type": checkpoint_data.get("current_ablation_type"),
                    "processing_stats": checkpoint_data.get("processing_stats", {})
                }
                
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh")
            return None
    
    def should_save_checkpoint(self, test_cases_processed_since_last: int) -> bool:
        """Determine if checkpoint should be saved based on adaptive interval."""
        return test_cases_processed_since_last >= self._adaptive_interval
    
    def save_test_case_results(self, file_id: str, ablation_type: str, test_case_data: Dict) -> None:
        """Save test case results grouped by file_id with both ablation types in one file."""
        import json
        
        if not test_case_data:
            return
            
        result_file = self.results_dir / f"{file_id}_results.json"
        
        try:
            # Load existing data if file exists
            existing_data = {}
            if result_file.exists():
                with open(result_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Initialize structure if needed
            if ablation_type not in existing_data:
                existing_data[ablation_type] = []
            
            # Add the new test case data
            test_case_entry = {
                "test_case_id": test_case_data.get('test_case_id', ''),
                "test_case_name": test_case_data.get('test_case_name', ''),
                "metrics": test_case_data.get('metrics', {})
            }
            existing_data[ablation_type].append(test_case_entry)
            
            # Write back to file
            with open(result_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Failed to save results for file {file_id}: {e}")
    
    def load_all_results(self) -> Dict[str, List]:
        """Load all saved test case results by ablation type from file-grouped structure."""
        import json
        
        all_results = {"ablation_1": [], "ablation_2": []}
        
        try:
            for result_file in self.results_dir.glob("*_results.json"):
                # Parse filename: file_id_results.json (e.g., 100_results.json)
                file_id = result_file.stem.replace('_results', '')
                
                with open(result_file, 'r') as f:
                    file_data = json.load(f)
                
                # Process each ablation type in the file
                for ablation_type in ["ablation_1", "ablation_2"]:
                    if ablation_type in file_data:
                        for test_case_data in file_data[ablation_type]:
                            # Reconstruct full result data
                            result_data = {
                                'test_case_id': test_case_data.get('test_case_id', ''),
                                'test_case_name': test_case_data.get('test_case_name', ''),
                                'ablation_type': ablation_type,
                                'metrics': test_case_data.get('metrics', {}),
                                'file_id': file_id
                            }
                            all_results[ablation_type].append(result_data)
                            
        except Exception as e:
            print(f"Warning: Failed to load some results: {e}")
        
        return all_results
    
    def clear_checkpoint(self):
        """Remove checkpoint and temporary result files after successful completion."""
        with self._lock:
            # Remove checkpoint files
            for file_path in [
                self.checkpoint_file,
                self.checkpoint_file.with_suffix('.json.gz.checksum')
            ]:
                if file_path.exists():
                    file_path.unlink()
            
            # Remove temporary results directory
            if self.results_dir.exists():
                import shutil
                shutil.rmtree(self.results_dir)
    
    def get_adaptive_interval(self) -> int:
        """Get current adaptive checkpoint interval."""
        return self._adaptive_interval


class ModelInitializer:
    """Handles initialization of models used in evaluation."""
    
    @staticmethod
    def initialize_nvembed(config: Dict):
        """Initialize NV-Embed model for embeddings."""
        global tokenizer_nv, model_nv, device_embed
        
        # Use local path like MPII
        model_path = Path(config['models']['nv_embed_path'])
        use_cuda = config.get('advanced', {}).get('use_cuda', True)
        device_embed = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        
        print(f"Initializing NV-Embed model from: {model_path}")
        print(f"Device: {device_embed}")
        
        tokenizer_nv = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_nv = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        tokenizer_nv.padding_side = "right"
        model_nv.eval()
        model_nv.to(device_embed)
        
        print("NV-Embed model initialized successfully")
    
    @staticmethod
    def initialize_sat(config: Dict):
        """Initialize SaT model for text segmentation."""
        global sat_adapted
        
        # Use sat_model from config like MPII
        model_name = config['models']['sat_model']
        
        print(f"Initializing SaT model: {model_name}")
        
        sat_adapted = SaT(model_name)
        
        print("SaT model initialized successfully")


class TextProcessor:
    """Handles text processing operations."""
    
    @staticmethod
    def sat_segmenter(text: str) -> List[str]:
        """
        Segment text into sentences using SAT model.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentence segments
        """
        global sat_adapted
        if sat_adapted is None:
            raise RuntimeError("SaT model not initialized. Call ModelInitializer.initialize_sat() first.")
        
        # Preprocessing pipeline
        text = contractions.fix(text)
        text = TextProcessor._remove_punctuation(text)
        text = TextProcessor._fix_punctuation_spacing(text)
        
        # Segmentation
        sentences = sat_adapted.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation except apostrophes."""
        return text.translate(str.maketrans('', '', ''.join(PUNCTUATIONS)))
    
    @staticmethod
    def _fix_punctuation_spacing(text: str) -> str:
        """Ensure proper spacing after punctuation."""
        return re.sub(r'([.!?])(?=[^\s])', r'\1 ', text)
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for evaluation."""
        if not text:
            return ""
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove punctuation except apostrophes
        text = ''.join(char for char in text if char not in PUNCTUATIONS)
        
        return text.lower()
    
    @staticmethod
    def get_segmenter_function(segmenter_name: str):
        """Get segmenter function by name."""
        if segmenter_name == "sat":
            return TextProcessor.sat_segmenter
        else:
            raise ValueError(f"Unknown segmenter: {segmenter_name}")


class EmbeddingGenerator:
    """Handles embedding generation for texts."""
    
    @staticmethod
    def nv_embed_embedding_fn(
        texts: List[str], 
        instruction: str = "", 
        model=None,
        batch_size: int = 8, 
        max_length: int = 32768
    ) -> torch.Tensor:
        """
        Generate normalized embeddings using NV-embed model.
        
        Args:
            texts: List of input texts
            instruction: Optional instruction for embedding
            model: Model to use (defaults to global model)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            
        Returns:
            Tensor of normalized embeddings
        """
        global model_nv
        
        if model is None:
            model = model_nv
            
        if model is None:
            raise RuntimeError("NV-embed model not initialized. Call ModelInitializer.initialize_nvembed() first.")
        
        device = next(model.parameters()).device
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_np = model.encode(batch, instruction=instruction, max_length=max_length)
            embeddings = torch.tensor(embeddings_np, device=device, dtype=torch.float)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)


class AblationUtils:
    """Utilities specific to ablation studies."""
    
    @staticmethod
    def load_test_case(json_file_path: str) -> Dict:
        """Load a single test case from JSON file."""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Error loading test case from {json_file_path}: {e}")
    
    @staticmethod
    def extract_test_cases(data_dir: str) -> List[Dict]:
        """Extract all test cases from JSON files in directory - matches MPII pattern."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        json_files = sorted(data_path.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in: {data_dir}")
        
        all_test_cases = []
        for json_file in json_files:
            # Load and parse JSON file (MPII pattern)
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            reference_text = data.get("ground_truth", "")
            if not reference_text:
                print(f"Warning: No ground_truth in file {json_file}")
                continue
            
            file_id = json_file.stem
            
            # Extract test cases (MPII pattern)
            test_cases = []
            for category in data.get("categories", []):
                for tc in category.get("test_cases", []):
                    test_cases.append((
                        tc.get("id", ""),
                        tc.get("name", ""),
                        tc.get("description", "")
                    ))
            
            # Convert to our format for processing
            for test_id, test_name, generated_text in test_cases:
                if not generated_text:
                    continue
                
                test_case_data = {
                    'test_case_id': test_id,
                    'test_case_name': test_name,
                    'reference': reference_text,
                    'generated': generated_text,
                    'file_id': file_id  # For grouping results
                }
                all_test_cases.append(test_case_data)
        
        print(f"Loaded {len(all_test_cases)} test cases from {len(json_files)} files in {data_dir}")
        return all_test_cases
    
    @staticmethod
    def compute_ablation_2_metrics(gas: float, las: float, nas_d: float, 
                                  nas_l: float, nas: float) -> Dict[str, float]:
        """
        Compute ablation 2 metrics with scaled combinations.
        Based on MPII addition script logic.
        """
        def compute_sas_cas_scaled(sas, cas):
            """Compute scaled combination of SAS and CAS."""
            if cas <= 0:
                return 0.0
            val = sas - (1 - cas)
            return (val / cas) if (val > 0) else 0.0
        
        def compute_sas_nas_scaled(sas, nas):
            """Compute scaled combination of SAS and NAS."""
            if sas < nas:
                numerator = sas - (1 - nas)
                denominator = nas
            else:
                numerator = nas - (1 - sas)
                denominator = sas
            return (numerator / denominator) if (numerator > 0 and denominator != 0) else 0.0
        
        # Compute scaled metrics
        nas_plus_cas_scaled = compute_sas_cas_scaled(nas, las)
        gas_plus_cas_scaled = compute_sas_cas_scaled(gas, las)
        gas_plus_nas_l_scaled = compute_sas_nas_scaled(gas, nas_l)
        gas_plus_nas_d_scaled = compute_sas_nas_scaled(gas, nas_d)
        gas_plus_nas_scaled = compute_sas_nas_scaled(gas, nas)
        gas_cas_s_plus_nas_d = compute_sas_nas_scaled(gas_plus_cas_scaled, nas_d)
        gas_cas_s_plus_nas_l = compute_sas_nas_scaled(gas_plus_cas_scaled, nas_l)
        
        # Final VCS score (combination of all scaled metrics)
        vcs_score = compute_sas_nas_scaled(gas_plus_cas_scaled, nas)
        
        return {
            "GAS": gas,
            "LAS": las,
            "NAS-D": nas_d,
            "NAS-L": nas_l,
            "NAS": nas,
            "NAS\n+LAS(S)": nas_plus_cas_scaled,
            "GAS\n+LAS(S)": gas_plus_cas_scaled,
            "GAS\n+NAS-L(S)": gas_plus_nas_l_scaled,
            "GAS\n+NAS-D(S)": gas_plus_nas_d_scaled,
            "GAS\n+NAS(S)": gas_plus_nas_scaled,
            "GAS\n+LAS(S)\n+NAS-D(S)": gas_cas_s_plus_nas_d,
            "GAS\n+LAS(S)\n+NAS-L(S)": gas_cas_s_plus_nas_l,
            "GAS\n+LAS(S)\n+(NAS-D\n+NAS-L)(S)": vcs_score
        }
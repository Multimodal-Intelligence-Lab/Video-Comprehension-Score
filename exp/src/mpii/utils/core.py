"""
Common classes and constants used across MPII evaluation scripts.

This module contains classes and constants that are identical across all evaluation scripts,
extracted to eliminate code duplication and improve maintainability.
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

# ============================================================================
# CONSTANTS
# ============================================================================

# Punctuation handling - exclude apostrophes for text processing
PUNCTUATIONS = set(string.punctuation) - {"'"}

# Output formatting
DECIMAL_PRECISION = 3

# Metrics ordering for consistent output
COMPARISON_METRICS_ORDER = [
    "BLEU-1", "BLEU-4", "METEOR", 
    "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum", "VCS"
]


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
        """
        Load configuration from YAML file with automatic base config merging.
        
        If _base.yaml exists in the same directory, it will be loaded first
        and the experiment config will be merged on top of it.
        """
        try:
            config_path = Path(config_path)
            config_dir = config_path.parent
            base_config_path = config_dir / "_base.yaml"
            
            # Load base config if it exists
            base_config = {}
            if base_config_path.exists():
                with open(base_config_path, 'r', encoding='utf-8') as f:
                    base_config = yaml.safe_load(f) or {}
            
            # Load experiment-specific config
            with open(config_path, 'r', encoding='utf-8') as f:
                exp_config = yaml.safe_load(f) or {}
            
            # Merge base + experiment configs (experiment takes precedence)
            if base_config:
                merged_config = deep_merge(base_config, exp_config)
            else:
                merged_config = exp_config
            
            ConfigLoader._validate_config(merged_config)
            return merged_config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate required configuration fields."""
        required_sections = ['models', 'paths', 'vcs', 'processing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate essential paths
        if 'nv_embed_path' not in config['models']:
            raise ValueError("Missing required field: models.nv_embed_path")
        
        # Validate LCT values
        if 'lct' not in config['vcs']:
            raise ValueError("Missing required field: vcs.lct")
        
        lct_values = config['vcs']['lct']
        if not isinstance(lct_values, list) or not all(isinstance(x, int) and x >= 0 for x in lct_values):
            raise ValueError("vcs.lct must be a list of non-negative integers")


class CheckpointManager:
    """Enhanced checkpoint manager optimized for large-scale processing.
    
    Features:
    - Memory-efficient storage (metadata only, not full results)
    - Thread-safe operations with locks
    - Integrity validation with checksums
    - Configuration consistency validation
    - Compressed storage format
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
        processed_files: List[str], 
        failed_files: List[str] = None,
        processing_stats: Dict = None
    ) -> None:
        """Save lightweight checkpoint with metadata only.
        
        Args:
            processed_files: List of successfully processed file paths
            failed_files: List of files that failed processing (optional)
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
                avg_time_per_file = processing_stats.get('avg_time_per_file', 60)
                # Adjust interval: save every 2-3 minutes of processing time
                target_interval_time = 150  # 2.5 minutes
                self._adaptive_interval = max(3, int(target_interval_time / avg_time_per_file))
            
            self._last_save_time = current_time
            
            checkpoint_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "processed_files": processed_files,
                "failed_files": failed_files or [],
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
                    "processed_files": checkpoint_data.get("processed_files", []),
                    "failed_files": checkpoint_data.get("failed_files", []),
                    "processing_stats": checkpoint_data.get("processing_stats", {})
                }
                
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh")
            return None
    
    def should_save_checkpoint(self, files_processed_since_last: int) -> bool:
        """Determine if checkpoint should be saved based on adaptive interval."""
        return files_processed_since_last >= self._adaptive_interval
    
    def save_file_results(self, file_path: str, results: List) -> None:
        """Save individual file results separately for memory efficiency."""
        import json
        
        if not results:
            return
            
        result_file = self.results_dir / f"{Path(file_path).stem}_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            if hasattr(result, '__dict__'):
                serializable_results.append(result.__dict__)
            else:
                serializable_results.append(result)
        
        try:
            with open(result_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save results for {file_path}: {e}")
    
    def load_all_results(self) -> List:
        """Load all saved file results."""
        import json
        from dataclasses import dataclass
        from typing import Dict
        
        @dataclass
        class EvaluationResult:
            """Container for evaluation metrics from a single test case."""
            test_case_id: str
            test_case_name: str
            metrics: Dict[str, float]
        
        @dataclass 
        class ComparisonResult:
            """Container for evaluation metrics from a single comparison."""
            ref_author: str
            other_author: str
            index_str: str
            metrics: Dict[str, float]
        
        all_results = []
        
        try:
            for result_file in self.results_dir.glob("*_results.json"):
                with open(result_file, 'r') as f:
                    file_results_raw = json.load(f)
                    
                    # Auto-detect result type and convert dictionaries back to appropriate objects
                    file_results = []
                    for result_dict in file_results_raw:
                        # Check if this looks like ComparisonResult (authors script)
                        if 'ref_author' in result_dict and 'other_author' in result_dict:
                            result = ComparisonResult(
                                ref_author=result_dict.get('ref_author', ''),
                                other_author=result_dict.get('other_author', ''),
                                index_str=result_dict.get('index_str', ''),
                                metrics=result_dict.get('metrics', {})
                            )
                        else:
                            # Default to EvaluationResult (comparison script)
                            result = EvaluationResult(
                                test_case_id=result_dict.get('test_case_id', ''),
                                test_case_name=result_dict.get('test_case_name', ''),
                                metrics=result_dict.get('metrics', {})
                            )
                        file_results.append(result)
                    
                    all_results.append(file_results)
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
    """Handles initialization of deep learning models."""
    
    # Class-level storage for models
    _sat_model = None
    _tokenizer_nv = None
    _model_nv = None
    _device_embed = None
    
    @classmethod
    def initialize_sat(cls) -> None:
        """Initialize SAT model for sentence segmentation."""
        import sys
        # Get the calling module to set the global variable there
        frame = sys._getframe(1)
        calling_globals = frame.f_globals
        
        cls._sat_model = SaT("sat-12l-sm")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._sat_model.half().to(device)
        calling_globals['sat_adapted'] = cls._sat_model

    @classmethod
    def initialize_nvembed(cls, config: Dict) -> None:
        """Initialize NV-embed model for semantic embeddings."""
        import sys
        # Get the calling module to set the global variables there
        frame = sys._getframe(1)
        calling_globals = frame.f_globals
        
        model_path = Path(config['models']['nv_embed_path'])
        use_cuda = config.get('advanced', {}).get('use_cuda', True)
        cls._device_embed = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        cls._tokenizer_nv = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cls._model_nv = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        cls._tokenizer_nv.padding_side = "right"
        cls._model_nv.eval()
        cls._model_nv.to(cls._device_embed)
        
        # Set in calling module's globals
        calling_globals['tokenizer_nv'] = cls._tokenizer_nv
        calling_globals['model_nv'] = cls._model_nv
        calling_globals['device_embed'] = cls._device_embed
    
    @classmethod
    def get_sat_model(cls):
        """Get the initialized SAT model."""
        return cls._sat_model
    
    @classmethod
    def get_nv_model(cls):
        """Get the initialized NV-embed model."""
        return cls._model_nv


class TextProcessor:
    """Handles text preprocessing and segmentation."""
    
    @staticmethod
    def sat_segmenter(text: str) -> List[str]:
        """
        Segment text into sentences using SAT model.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of sentence segments
        """
        import sys
        # Get sat_adapted from calling module first, then fallback to class storage
        frame = sys._getframe(1)
        calling_globals = frame.f_globals
        sat_adapted = calling_globals.get('sat_adapted')
        
        if sat_adapted is None:
            sat_adapted = ModelInitializer.get_sat_model()
        
        if sat_adapted is None:
            raise RuntimeError("SAT model not initialized. Call ModelInitializer.initialize_sat() first.")
        
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


class EmbeddingGenerator:
    """Handles generation of semantic embeddings."""
    
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
        if model is None:
            import sys
            # Get model_nv from calling module first, then fallback to class storage
            frame = sys._getframe(1)
            calling_globals = frame.f_globals
            model = calling_globals.get('model_nv')
            
            if model is None:
                model = ModelInitializer.get_nv_model()
            
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
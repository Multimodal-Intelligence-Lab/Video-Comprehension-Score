"""
Common classes and functions for VATEX-EVAL evaluation scripts.

This module contains classes and functions that are shared across VATEX-EVAL
evaluation scripts, including segmenter functions and infrastructure components.
"""

import torch
import torch.nn.functional as F
import re
import os
import time
import pickle
import yaml
import string
import json
import gzip
import hashlib
import threading
import shutil
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.stats import kendalltau, spearmanr
import ast
import nltk

from nltk.corpus import stopwords

# ============================================================================
# CONSTANTS
# ============================================================================

# Punctuation characters for text processing
PUNCTUATIONS = list(string.punctuation)

# Decimal precision for metric outputs
DECIMAL_PRECISION = 4

# VCS metrics utilities for dynamic metric generation

class VCSMetricsGenerator:
    """Utility class for generating VCS metric names and configurations."""
    
    @staticmethod
    def generate_metric_names(chunk_sizes: List[int], lct_values: List[int]) -> List[str]:
        """
        Generate VCS metric names in VCS_C{chunk}_LCT{n} format.
        
        Args:
            chunk_sizes: List of chunk sizes to use
            lct_values: List of LCT values to use
            
        Returns:
            List of metric names in sorted order
        """
        metric_names = []
        for chunk_size in sorted(chunk_sizes):
            for lct in sorted(lct_values):
                metric_names.append(f"VCS_C{chunk_size}_LCT{lct}")
        return metric_names
    
    @staticmethod
    def parse_metric_name(metric_name: str) -> tuple:
        """
        Parse VCS metric name to extract chunk size and LCT value.
        
        Args:
            metric_name: Metric name in VCS_C{chunk}_LCT{n} format
            
        Returns:
            Tuple of (chunk_size, lct_value) or None if invalid format
        """
        import re
        match = re.match(r'VCS_C(\d+)_LCT(\d+)', metric_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None
    
    @staticmethod
    def get_default_vcs_config() -> Dict:
        """Get default VCS configuration."""
        return {
            "chunk_size": [1],
            "lct_values": [0],
            "context_cutoff_value": 0.6,
            "context_window_control": 4.0,
            "return_all_metrics": True,
            "return_internals": False
        }

# Default segmenter function for VATEX-EVAL (single segmenter)
DEFAULT_SEGMENTER_FUNCTION = "segmenter_punc_stop"

# Default n_refs values for VATEX-EVAL
DEFAULT_N_REFS = [1, 9]

# ============================================================================
# CONFIGURATION DEFAULTS
# ============================================================================

# VCS computation default parameters
VCS_DEFAULTS = {
    "chunk_size": 1,
    "context_cutoff_value": 0.6,
    "context_window_control": 4.0,
    "return_all_metrics": True,
    "return_internals": False,
}

# Embedding model default parameters
EMBEDDING_DEFAULTS = {
    "batch_size": 8,
    "max_length": 32768,
    "instruction": "",
}

# Processing default parameters
PROCESSING_DEFAULTS = {
    "max_workers": 4,
    "checkpoint_interval": 100,
    "resume_from_checkpoint": True,
}

# Output formatting defaults
OUTPUT_DEFAULTS = {
    "decimal_precision": 4,
    "save_individual_results": True,
    "save_correlation_summary": True,
    "save_detailed_scores": True,
}


class ConfigLoader:
    """Handles loading and validation of configuration files."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            ConfigLoader._validate_config(config)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate required configuration fields."""
        required_sections = ['models', 'paths', 'vatex_eval', 'processing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate essential paths
        if 'nv_embed_path' not in config['models']:
            raise ValueError("Missing required field: models.nv_embed_path")
        
        # Validate VATEX-EVAL specific settings
        if 'data_dir' not in config['vatex_eval']:
            raise ValueError("Missing required field: vatex_eval.data_dir")
        
        if 'use_n_refs' not in config['vatex_eval']:
            raise ValueError("Missing required field: vatex_eval.use_n_refs")
        
        # Validate use_n_refs values
        use_n_refs = config['vatex_eval']['use_n_refs']
        if not isinstance(use_n_refs, list) or not all(isinstance(x, int) and x >= 1 for x in use_n_refs):
            raise ValueError("vatex_eval.use_n_refs must be a list of positive integers")
        
        # Note: segmenter_functions is now optional as we use a default single segmenter
        
        # Validate LCT values
        if 'lct_values' not in config['vcs']:
            raise ValueError("Missing required field: vcs.lct_values")
        
        lct_values = config['vcs']['lct_values']
        if not isinstance(lct_values, list) or not all(isinstance(x, int) and x >= 0 for x in lct_values):
            raise ValueError("vcs.lct_values must be a list of non-negative integers")
        
        # Validate chunk sizes
        if 'chunk_size' not in config['vcs']:
            raise ValueError("Missing required field: vcs.chunk_size")
        
        chunk_sizes = config['vcs']['chunk_size']
        if not isinstance(chunk_sizes, list) or not all(isinstance(x, int) and x >= 1 for x in chunk_sizes):
            raise ValueError("vcs.chunk_size must be a list of positive integers")


class CheckpointManager:
    """Enhanced checkpoint manager optimized for VATEX-EVAL processing."""
    
    def __init__(self, checkpoint_dir: str, experiment_id: str, config_hash: str = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.config_hash = config_hash
        self.checkpoint_file = self.checkpoint_dir / f"checkpoint_{experiment_id}.json.gz"
        # Base results directory - will be made n_refs-specific when needed
        self.base_results_dir = self.checkpoint_dir / f"results_{experiment_id}"
        self.current_results_dir = None  # Will be set per n_refs configuration
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._last_save_time = 0
        self._adaptive_interval = 100  # Start with 100 candidates, adapt based on processing speed
    
    def set_current_config(self, n_refs: int):
        """Set the current n_refs configuration and create corresponding temp directory."""
        self.current_results_dir = self.base_results_dir / f"nrefs_{n_refs}"
        self.current_results_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def results_dir(self):
        """Get the current results directory for the active n_refs configuration."""
        if self.current_results_dir is None:
            raise RuntimeError("No current configuration set. Call set_current_config() first.")
        return self.current_results_dir
    
    def get_existing_result_files(self, n_refs: int) -> set:
        """Get set of existing result files for a given n_refs configuration."""
        existing_files = set()
        
        # Check the specific temp results directory for this n_refs configuration
        config_results_dir = self.base_results_dir / f"nrefs_{n_refs}"
        if config_results_dir.exists():
            for json_file in config_results_dir.glob("*.json"):
                # Extract video_id from filename (remove .json extension)
                video_id = json_file.stem
                existing_files.add(video_id)
        
        return existing_files
    
    def count_processed_candidates_from_files(self, existing_files: set, video_ids: list) -> int:
        """Count how many candidates have been processed based on existing files."""
        processed_count = 0
        for video_id in video_ids:
            if video_id in existing_files:
                processed_count += 1
        return processed_count
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate SHA-256 checksum for integrity validation."""
        return hashlib.sha256(data).hexdigest()
    
    def _generate_config_hash(self, config: Dict) -> str:
        """Generate a hash of the configuration for consistency validation."""
        config_copy = config.copy()
        config_copy.pop('experiment', None)  # Remove experiment-specific fields
        config_str = json.dumps(config_copy, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def save_checkpoint(
        self, 
        processed_candidates: int,
        total_candidates: int,
        current_segmenter: str,
        current_n_refs: int,
        processing_stats: Dict = None
    ) -> None:
        """Save checkpoint with VATEX-EVAL specific state."""
        
        with self._lock:
            current_time = time.time()
            if processing_stats and self._last_save_time > 0:
                time_since_last = current_time - self._last_save_time
                avg_time_per_candidate = processing_stats.get('avg_time_per_candidate', 1.0)
                # Adjust interval: save every 2-3 minutes of processing time
                target_interval_time = 180  # 3 minutes
                self._adaptive_interval = max(50, int(target_interval_time / avg_time_per_candidate))
            
            self._last_save_time = current_time
            
            checkpoint_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "processed_candidates": processed_candidates,
                "total_candidates": total_candidates,
                "current_segmenter": current_segmenter,
                "current_n_refs": current_n_refs,
                "processing_stats": processing_stats or {},
                "adaptive_interval": self._adaptive_interval
            }
            
            # Atomic write using temporary file with compression
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            try:
                json_data = json.dumps(checkpoint_data, indent=2)
                compressed_data = gzip.compress(json_data.encode('utf-8'))
                
                checksum = self._calculate_checksum(compressed_data)
                
                with open(temp_file, 'wb') as f:
                    f.write(compressed_data)
                
                checksum_file = temp_file.with_suffix('.checksum')
                with open(checksum_file, 'w') as f:
                    f.write(checksum)
                
                temp_file.replace(self.checkpoint_file)
                checksum_file.replace(self.checkpoint_file.with_suffix('.json.gz.checksum'))
                
            except Exception as e:
                for temp in [temp_file, temp_file.with_suffix('.checksum')]:
                    if temp.exists():
                        temp.unlink()
                raise e
    
    def load_checkpoint(self, config: Dict = None) -> Optional[Dict]:
        """Load and validate checkpoint."""
        if not self.checkpoint_file.exists():
            return None
        
        checksum_file = self.checkpoint_file.with_suffix('.json.gz.checksum')
        
        try:
            with self._lock:
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
                    with open(self.checkpoint_file, 'rb') as f:
                        data = f.read()
                    print("Warning: Loading checkpoint without integrity validation")
                
                try:
                    json_data = gzip.decompress(data).decode('utf-8')
                except gzip.BadGzipFile:
                    json_data = data.decode('utf-8')
                
                checkpoint_data = json.loads(json_data)
                
                if config and self.config_hash:
                    current_config_hash = self._generate_config_hash(config)
                    if checkpoint_data.get('config_hash') != current_config_hash:
                        print("Warning: Configuration has changed since checkpoint, starting fresh")
                        return None
                
                if 'adaptive_interval' in checkpoint_data:
                    self._adaptive_interval = checkpoint_data['adaptive_interval']
                
                return checkpoint_data
                
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh")
            return None
    
    def should_save_checkpoint(self, candidates_processed_since_last: int) -> bool:
        """Determine if checkpoint should be saved based on adaptive interval."""
        return candidates_processed_since_last >= self._adaptive_interval
    
    def clear_checkpoint(self):
        """Remove checkpoint and temporary result files after successful completion."""
        with self._lock:
            for file_path in [
                self.checkpoint_file,
                self.checkpoint_file.with_suffix('.json.gz.checksum')
            ]:
                if file_path.exists():
                    file_path.unlink()
            
            # Clean up all n_refs temp directories
            if self.base_results_dir.exists():
                shutil.rmtree(self.base_results_dir)
    
    def get_adaptive_interval(self) -> int:
        """Get current adaptive checkpoint interval."""
        return self._adaptive_interval


class ModelInitializer:
    """Handles initialization of deep learning models."""
    
    # Class-level storage for models
    _tokenizer_nv = None
    _model_nv = None
    _device_embed = None
    
    @classmethod
    def initialize_nvembed(cls, config: Dict) -> None:
        """Initialize NV-embed model for semantic embeddings."""
        import sys
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
        
        calling_globals['tokenizer_nv'] = cls._tokenizer_nv
        calling_globals['model_nv'] = cls._model_nv
        calling_globals['device_embed'] = cls._device_embed
    
    @classmethod
    def get_nv_model(cls):
        """Get the initialized NV-embed model."""
        return cls._model_nv


class TextProcessor:
    """Handles text preprocessing and segmentation."""
    
    @staticmethod
    def segmenter_punc_stop(text: str) -> List[str]:
        """
        Removes punctuation and splits text into words.
        Additionally, removes English stopwords (NLTK or built-in fallback).
        Does not expand contractions.
        
        Args:
            text (str): The input text.
            
        Returns:
            List[str]: A list of words extracted from the text.
        """
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        words = text.split()
        
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word.lower() not in stop_words]
        return words
    
    @staticmethod
    def get_segmenter_function(segmenter_name: str):
        """Get segmenter function by name."""
        if segmenter_name == 'segmenter_punc_stop':
            return TextProcessor.segmenter_punc_stop
        else:
            raise ValueError(f"Unknown segmenter function: {segmenter_name}. Only 'segmenter_punc_stop' is supported.")


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


class VATEXEvalUtils:
    """Utility functions specific to VATEX-EVAL processing."""
    
    @staticmethod
    def load_vatex_data(json_path: str) -> Dict:
        """Load VATEX-EVAL data from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load VATEX-EVAL data from {json_path}: {e}")
    
    @staticmethod
    def extract_candidates_and_references(data: Dict) -> Tuple[List[str], List[List[str]], np.ndarray, List[str]]:
        """
        Extract candidates, references, and human scores from VATEX-EVAL data.
        
        Returns:
            Tuple of (candidates, references, human_scores_array, video_ids)
        """
        cands_all = []
        refs_all = []
        human_scores_all = []
        video_ids_all = []
        
        for video_id, video_data in data.items():
            video_ids_all.extend([video_id] * len(video_data['cands']))
            cands_all.extend(video_data['cands'])
            
            # Each candidate gets a list of references
            for _ in range(len(video_data['cands'])):
                refs_all.append(video_data['refs'][0])  # Each ref is a list of reference captions
            
            # Parse scores from string to list
            scores_matrix = ast.literal_eval(video_data['scores'])
            for score_row in scores_matrix:
                human_scores_all.append(score_row)  # 3 annotator scores per candidate
        
        human_scores_array = np.array(human_scores_all)
        
        return cands_all, refs_all, human_scores_array, video_ids_all
    
    @staticmethod
    def limit_references(refs_all: List[List[str]], use_n_refs: int) -> List[List[str]]:
        """Limit references to specified number."""
        refs_limited = []
        for ref_list in refs_all:
            refs_limited.append(ref_list[:use_n_refs])
        return refs_limited
    
    @staticmethod
    def compute_correlation_uniquehuman(pred: np.ndarray, all_human_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute correlation between metric scores and human judgments.
        
        Args:
            pred: Predicted scores array
            all_human_scores: Human scores array (n_samples, n_annotators)
            
        Returns:
            Tuple of (kendall_tau, spearman_correlation)
        """
        num_workers = 3  # 3 annotators in VATEX-EVAL
        
        # Round predictions to 4 decimal places (like EMScore)
        pred = np.around(pred, decimals=4)
        
        spearman = 0
        spearman_significant = True
        for worker_i in range(num_workers):
            tmp, p_value = spearmanr(pred, all_human_scores[:, worker_i])
            # P-value validation - warn if not significant but continue
            if p_value >= 0.01:
                print(f"Warning: Spearman correlation not significant for worker {worker_i+1}: p-value={p_value:.4f}")
                spearman_significant = False
            spearman += tmp
        spearman /= num_workers
        spearman = np.around(spearman, decimals=4)

        kendalltau_val = 0
        kendall_significant = True
        for worker_i in range(num_workers):
            tmp, p_value = kendalltau(pred, all_human_scores[:, worker_i])
            # P-value validation - warn if not significant but continue
            if p_value >= 0.01:
                print(f"Warning: Kendall correlation not significant for worker {worker_i+1}: p-value={p_value:.4f}")
                kendall_significant = False
            kendalltau_val += tmp
        kendalltau_val /= num_workers
        kendalltau_val = np.around(kendalltau_val, decimals=4)

        # Print correlation values with significance indicators
        kendall_status = "✓" if kendall_significant else "⚠"
        spearman_status = "✓" if spearman_significant else "⚠"
        print('kendall: {} {}, spear: {} {}'.format(kendalltau_val, kendall_status, spearman, spearman_status))
        
        return kendalltau_val, spearman
    
    @staticmethod
    def generate_individual_results_path(base_output_dir: str, n_refs: int) -> str:
        """Generate standardized individual results path for new structure."""
        return str(Path(base_output_dir) / "individual_results" / f"{n_refs}ref")
    
    @staticmethod
    def generate_aggregated_results_path(base_output_dir: str) -> str:
        """Generate standardized aggregated results path for new structure."""
        return str(Path(base_output_dir) / "aggregated_results")
    
    @staticmethod
    def generate_file_suffix(n_refs: int, metric_config: str = "") -> str:
        """Generate standardized file suffix for new structure."""
        if metric_config:
            return f"{n_refs}refs_{metric_config}"
        return f"{n_refs}refs"
    
    @staticmethod
    def generate_hierarchical_results_structure(base_output_dir: str, n_refs_list: List[int]) -> Dict[str, str]:
        """
        Generate complete directory structure for hierarchical VCS results.
        
        Args:
            base_output_dir: Base output directory
            n_refs_list: List of n_refs values to create directories for
            
        Returns:
            Dictionary mapping result types to paths
        """
        structure = {
            "base": base_output_dir,
            "individual_results": {},
            "aggregated_results": VATEXEvalUtils.generate_aggregated_results_path(base_output_dir)
        }
        
        for n_refs in n_refs_list:
            structure["individual_results"][n_refs] = VATEXEvalUtils.generate_individual_results_path(base_output_dir, n_refs)
        
        return structure
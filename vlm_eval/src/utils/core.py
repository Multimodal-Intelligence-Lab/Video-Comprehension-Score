"""
Common classes and utilities for VLM evaluation on CLIP-CC dataset.

This module contains classes and utilities for evaluating VLMs against CLIP-CC ground truth
using VCS scores with parallel processing and checkpointing capabilities.
"""

import torch
import torch.nn.functional as F
import re
import os
import time
import json
import pickle
import yaml
import string
import contractions
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel
from wtpsplit import SaT
from typing import List, Dict, Optional, Any, Tuple
from copy import deepcopy
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Global model instances
sat_adapted = None
model_nv = None
device_embed = None

# ============================================================================
# CONSTANTS
# ============================================================================

# Punctuation handling - exclude apostrophes for text processing
PUNCTUATIONS = set(string.punctuation) - {"'"}

# Output formatting
DECIMAL_PRECISION = 3

# VCS evaluation configurations - now dynamic based on config

@dataclass
class VLMEvaluationResult:
    """Container for VLM evaluation results."""
    id: str
    file_link: str
    vcs_scores: Dict[str, float]  # Dynamic VCS scores based on config
    model_name: str = ""

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
    """Handles loading and validation of configuration files for VLM evaluation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        Load configuration from YAML file for VLM evaluation.
        """
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
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate required configuration fields for VLM evaluation."""
        required_sections = ['models', 'paths', 'vcs', 'processing']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate essential paths
        if 'nv_embed_path' not in config['models']:
            raise ValueError("Missing required field: models.nv_embed_path")
        
        # Validate VCS parameters
        required_vcs_fields = ['chunk_sizes', 'lct_values']
        for field in required_vcs_fields:
            if field not in config['vcs']:
                raise ValueError(f"Missing required field: vcs.{field}")
        
        # Validate chunk_sizes and lct_values
        chunk_sizes = config['vcs']['chunk_sizes']
        lct_values = config['vcs']['lct_values']
        
        if not isinstance(chunk_sizes, list) or not all(isinstance(x, int) and x > 0 for x in chunk_sizes):
            raise ValueError("vcs.chunk_sizes must be a list of positive integers")
            
        if not isinstance(lct_values, list) or not all(isinstance(x, int) and x >= 0 for x in lct_values):
            raise ValueError("vcs.lct_values must be a list of non-negative integers")


class DataLoader:
    """Handles loading of CLIP-CC dataset and model predictions."""
    
    @staticmethod
    def load_clipcc_data(clipcc_dir: str) -> Dict[str, Dict]:
        """
        Load CLIP-CC ground truth data.
        
        Args:
            clipcc_dir: Path to CLIP-CC dataset directory
            
        Returns:
            Dictionary mapping ID to ground truth data
        """
        clipcc_path = Path(clipcc_dir)
        ground_truth = {}
        
        # Look for JSON files in the CLIP-CC directory
        for json_file in clipcc_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Handle both single objects and lists
                if isinstance(data, list):
                    for item in data:
                        if 'id' in item and 'summary' in item:
                            ground_truth[item['id']] = item
                elif isinstance(data, dict) and 'id' in data and 'summary' in data:
                    ground_truth[data['id']] = data
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        if not ground_truth:
            raise ValueError(f"No valid CLIP-CC data found in {clipcc_dir}")
            
        print(f"Loaded {len(ground_truth)} CLIP-CC ground truth entries")
        return ground_truth
    
    @staticmethod
    def load_model_predictions(models_dir: str) -> Dict[str, Dict]:
        """
        Load model predictions from JSON files.
        
        Args:
            models_dir: Path to models predictions directory
            
        Returns:
            Dictionary mapping model name to predictions dict (id -> prediction)
        """
        models_path = Path(models_dir)
        all_models = {}
        
        for json_file in models_path.glob("*.json"):
            model_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                model_predictions = {}
                # Handle both single objects and lists
                if isinstance(data, list):
                    for item in data:
                        if 'id' in item and ('summary' in item or 'description' in item):
                            # Use 'summary' if available, otherwise 'description'
                            prediction = item.get('summary', item.get('description', ''))
                            model_predictions[item['id']] = {
                                'id': item['id'],
                                'prediction': prediction,
                                'file_link': item.get('file_link', '')
                            }
                elif isinstance(data, dict):
                    for id_key, item in data.items():
                        if isinstance(item, dict) and ('summary' in item or 'description' in item):
                            prediction = item.get('summary', item.get('description', ''))
                            model_predictions[id_key] = {
                                'id': id_key,
                                'prediction': prediction,
                                'file_link': item.get('file_link', '')
                            }
                        elif isinstance(item, str):
                            # Handle format where dict maps ID -> description string directly
                            model_predictions[id_key] = {
                                'id': id_key,
                                'prediction': item,
                                'file_link': ''  # Will be filled from ground truth
                            }
                
                if model_predictions:
                    all_models[model_name] = model_predictions
                    print(f"Loaded {len(model_predictions)} predictions for model {model_name}")
                else:
                    print(f"Warning: No valid predictions found in {json_file}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        if not all_models:
            raise ValueError(f"No valid model predictions found in {models_dir}")
            
        return all_models


class VCSEvaluator:
    """Handles VCS evaluation for VLM predictions against CLIP-CC ground truth."""
    
    def __init__(self, config: Dict, checkpoint_manager=None, logger=None):
        """Initialize VCS evaluator with configuration."""
        self.config = config
        self.chunk_sizes = config['vcs']['chunk_sizes']
        self.lct_values = config['vcs']['lct_values']
        self.segmenter = TextProcessor.sat_segmenter
        self.embedding_fn = EmbeddingGenerator.nv_embed_embedding_fn
        self.checkpoint_manager = checkpoint_manager  # For incremental saving
        self.logger = logger  # For logging to file instead of terminal
        
    def evaluate_model(
        self, 
        model_name: str,
        model_predictions: Dict[str, Dict],
        ground_truth: Dict[str, Dict]
    ) -> List[VLMEvaluationResult]:
        """
        Evaluate a single model against ground truth using VCS with interval-based saving.
        
        Args:
            model_name: Name of the model being evaluated
            model_predictions: Model predictions dict (id -> prediction data)
            ground_truth: Ground truth dict (id -> ground truth data)
            
        Returns:
            List of evaluation results
        """
        from vcs import compute_vcs_score
        import time
        
        results = []
        
        # Find common IDs between predictions and ground truth
        common_ids = set(model_predictions.keys()) & set(ground_truth.keys())
        if not common_ids:
            if self.logger:
                self.logger.warning(f"No common IDs found between {model_name} and ground truth")
            else:
                print(f"Warning: No common IDs found between {model_name} and ground truth")
            return results
        
        # Filter out already processed files from checkpoint like mpii
        if self.checkpoint_manager and hasattr(self.checkpoint_manager, 'processed_files'):
            processed_files = self.checkpoint_manager.processed_files
            failed_files = self.checkpoint_manager.failed_files if hasattr(self.checkpoint_manager, 'failed_files') else set()
            remaining_ids = common_ids - processed_files - failed_files
        else:
            remaining_ids = common_ids
        
        total_files = len(remaining_ids)
        total_original = len(common_ids)
        
        if self.logger:
            if total_files < total_original:
                already_processed = total_original - total_files
                self.logger.info(f"Evaluating {total_files} remaining samples for model {model_name} ({already_processed} already completed)")
            else:
                self.logger.info(f"Evaluating {total_files} samples for model {model_name}")
        else:
            if total_files < total_original:
                already_processed = total_original - total_files
                print(f"Evaluating {total_files} remaining samples for model {model_name} ({already_processed} already completed)")
            else:
                print(f"Evaluating {total_files} samples for model {model_name}")
        
        # Initialize model results files if checkpoint manager is available
        if self.checkpoint_manager:
            self.checkpoint_manager.initialize_model_results(model_name)
        
        # Process each file with interval-based checkpointing like mpii
        processed_count = 0
        for id_key in sorted(remaining_ids):
            file_start_time = time.time()
            
            pred_data = model_predictions[id_key]
            gt_data = ground_truth[id_key]
            
            prediction = pred_data['prediction']
            reference = gt_data['summary']
            # Extract file_link directly from ground truth (clip_cc_dataset.json)
            file_link = gt_data.get('file_link', '')
            
            if not prediction.strip() or not reference.strip():
                if self.checkpoint_manager:
                    self.checkpoint_manager.track_file_failed(id_key, "Empty prediction or reference")
                # Use debug level for empty predictions to reduce log spam
                if self.logger:
                    self.logger.debug(f"Empty prediction or reference for ID {id_key}")
                else:
                    print(f"Warning: Empty prediction or reference for ID {id_key}")
                continue
            
            # Compute VCS scores for all chunk_size and LCT combinations
            vcs_scores = {}
            computation_failed = False
            
            for chunk_size in self.chunk_sizes:
                for lct in self.lct_values:
                    try:
                        vcs_results = compute_vcs_score(
                            reference_text=reference,
                            generated_text=prediction,
                            segmenter_fn=self.segmenter,
                            embedding_fn_las=self.embedding_fn,
                            embedding_fn_gas=self.embedding_fn,
                            chunk_size=chunk_size,
                            lct=lct,
                            context_cutoff_value=self.config['vcs']['context_cutoff_value'],
                            context_window_control=self.config['vcs']['context_window_control'],
                            return_all_metrics=True,
                            return_internals=False
                        )
                        vcs_scores[f"chunk{chunk_size}_lct{lct}"] = vcs_results.get("VCS", 0.0)
                    except Exception as e:
                        # Use debug level for VCS computation failures to reduce log spam
                        error_msg = f"VCS computation failed for {model_name}, ID {id_key}, chunk_size={chunk_size}, lct={lct}: {str(e)[:100]}"
                        if self.logger:
                            self.logger.debug(error_msg)  # Changed from warning to debug
                        else:
                            print(f"Warning: {error_msg}")
                        vcs_scores[f"chunk{chunk_size}_lct{lct}"] = 0.0
                        computation_failed = True
            
            # Track processing time and update statistics
            file_processing_time = time.time() - file_start_time
            
            if computation_failed:
                if self.checkpoint_manager:
                    self.checkpoint_manager.track_file_failed(id_key, "VCS computation failed")
            else:
                # Create result object with dynamic VCS scores
                result = VLMEvaluationResult(
                    id=id_key,
                    file_link=file_link,
                    vcs_scores=vcs_scores.copy(),
                    model_name=model_name
                )
                results.append(result)
                
                # Save result to tmp storage and track progress
                if self.checkpoint_manager:
                    self.checkpoint_manager.save_single_result(result)
                    self.checkpoint_manager.track_file_processed(id_key, file_processing_time)
                    
                    # Check if we should save checkpoint (interval-based like mpii)
                    if self.checkpoint_manager.should_save_checkpoint():
                        self.checkpoint_manager.save_interval_results_to_csv(model_name)
                        # Only log checkpoint saves, not individual file saves to reduce spam
                        if self.logger:
                            total_processed = self.checkpoint_manager.processing_stats.get("files_completed", 0)
                            self.logger.info(f"Checkpoint saved for {model_name} after {total_processed} files (interval: {self.checkpoint_manager._adaptive_interval})")
                        else:
                            print(f"Checkpoint saved for {model_name}")
            
            processed_count += 1
            
            # Progress logging every 20 files to reduce log spam
            if processed_count % 20 == 0 or processed_count == total_files:
                avg_time = self.checkpoint_manager.processing_stats.get("avg_time_per_file", 0) if self.checkpoint_manager else file_processing_time
                remaining = total_files - processed_count
                eta_minutes = (remaining * avg_time) / 60 if avg_time > 0 else 0
                progress_msg = f"Model {model_name} progress: {processed_count}/{total_files} ({processed_count/total_files*100:.1f}%) - ETA: {eta_minutes:.1f} min"
                if self.logger:
                    self.logger.info(progress_msg)
                else:
                    print(progress_msg)
        
        # Final save to CSV at the end of model evaluation
        if self.checkpoint_manager:
            self.checkpoint_manager.save_interval_results_to_csv(model_name)
            if self.logger:
                self.logger.info(f"Final results saved for model {model_name}")
            else:
                print(f"Final results saved for model {model_name}")
        
        return results


class CheckpointManager:
    """Checkpoint manager for VLM evaluation progress tracking - mirrors mpii approach."""
    
    def __init__(self, results_dir: str, experiment_id: str, config: Dict = None, logger=None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.checkpoint_file = self.results_dir / f"checkpoint_{experiment_id}.json.gz"
        self.checksum_file = self.results_dir / f"checkpoint_{experiment_id}.json.gz.checksum"
        # Create tmp checkpoint folder alongside .json.gz like mpii
        self.tmp_results_dir = self.results_dir / f"results_{experiment_id}"
        self.tmp_results_dir.mkdir(exist_ok=True)
        self.individual_results_dir = self.results_dir / "individual_results"
        self.individual_results_dir.mkdir(exist_ok=True)
        
        # Configuration and validation
        self.config = config
        self.config_hash = self._calculate_config_hash(config) if config else None
        self.logger = logger  # For logging to file instead of terminal
        
        # Adaptive interval mechanism (like mpii)
        self._adaptive_interval = config.get('processing', {}).get('checkpoint_interval', 5) if config else 5
        self._last_save_time = 0
        self._files_processed_since_last = 0
        self._processing_start_time = time.time()
        
        # File-level tracking (like mpii)
        self.processed_files = set()
        self.failed_files = set()
        
        # Processing statistics (like mpii)
        self.processing_stats = {
            "avg_time_per_file": 0.0,
            "total_elapsed": 0.0,
            "files_completed": 0,
            "files_failed": 0
        }
        
        # Create initial empty checkpoint immediately (like mpii)
        self._create_initial_checkpoint()
    
    def _calculate_config_hash(self, config: Dict) -> str:
        """Calculate hash of configuration for consistency validation."""
        if not config:
            return ""
        import hashlib
        import json
        # Create deterministic string representation
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def should_save_checkpoint(self, files_processed_since_last: int = None) -> bool:
        """Determine if checkpoint should be saved based on adaptive interval (like mpii)."""
        if files_processed_since_last is not None:
            self._files_processed_since_last = files_processed_since_last
        return self._files_processed_since_last >= self._adaptive_interval
    
    def _update_adaptive_interval(self) -> None:
        """Update adaptive interval based on processing speed (like mpii)."""
        if self.processing_stats["files_completed"] > 0 and self.processing_stats["avg_time_per_file"] > 0:
            # Target: checkpoint every 2.5 minutes (150 seconds)
            target_interval_time = 150
            optimal_interval = max(3, int(target_interval_time / self.processing_stats["avg_time_per_file"]))
            # Cap the interval at a reasonable maximum (like mpii does)
            self._adaptive_interval = min(optimal_interval, 100)
    
    def _create_initial_checkpoint(self) -> None:
        """Create initial empty checkpoint on startup (like mpii)."""
        if not self.checkpoint_file.exists():
            initial_checkpoint = {
                "version": "2.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "processed_models": [],
                "processed_files": list(self.processed_files),
                "failed_files": list(self.failed_files),
                "processing_stats": self.processing_stats.copy(),
                "adaptive_interval": self._adaptive_interval
            }
            self._save_checkpoint_with_checksum(initial_checkpoint)
    
    def _calculate_checksum(self, data: bytes) -> str:
        """Calculate MD5 checksum of data."""
        import hashlib
        return hashlib.md5(data).hexdigest()
    
    def _save_checkpoint_with_checksum(self, checkpoint_data: dict) -> None:
        """Save checkpoint with checksum like mpii."""
        import gzip
        import json
        
        # Create JSON and compress
        json_data = json.dumps(checkpoint_data, indent=2)
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        
        # Calculate checksum
        checksum = self._calculate_checksum(compressed_data)
        
        # Write checkpoint file
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                f.write(compressed_data)
            
            # Write checksum file
            checksum_temp = temp_file.with_suffix('.checksum')
            with open(checksum_temp, 'w') as f:
                f.write(checksum)
            
            # Atomic moves
            temp_file.replace(self.checkpoint_file)
            checksum_temp.replace(self.checksum_file)
            
        except Exception as e:
            # Cleanup on failure
            for temp in [temp_file, temp_file.with_suffix('.checksum')]:
                if temp.exists():
                    temp.unlink()
            raise e
        
    def save_checkpoint(self, processed_models: List[str], processing_stats: Dict = None) -> None:
        """Save checkpoint with enhanced structure like mpii."""
        # Update processing statistics
        if processing_stats:
            self.processing_stats.update(processing_stats)
        
        # Update adaptive interval based on current performance
        self._update_adaptive_interval()
        
        checkpoint_data = {
            "version": "2.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "experiment_id": self.experiment_id,
            "config_hash": self.config_hash,
            "processed_models": processed_models,
            "processed_files": list(self.processed_files),
            "failed_files": list(self.failed_files),
            "processing_stats": self.processing_stats.copy(),
            "adaptive_interval": self._adaptive_interval
        }
        
        self._save_checkpoint_with_checksum(checkpoint_data)
        self._files_processed_since_last = 0  # Reset counter after saving
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint with validation and state restoration like mpii."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            import gzip
            # Validate checksum if checksum file exists
            if self.checksum_file.exists():
                with open(self.checksum_file, 'r') as f:
                    expected_checksum = f.read().strip()
                
                with open(self.checkpoint_file, 'rb') as f:
                    data = f.read()
                
                actual_checksum = self._calculate_checksum(data)
                if actual_checksum != expected_checksum:
                    if self.logger:
                        self.logger.warning("Checkpoint integrity check failed, starting fresh")
                    else:
                        print("Warning: Checkpoint integrity check failed, starting fresh")
                    return None
            else:
                # Load without checksum validation (backward compatibility)
                with open(self.checkpoint_file, 'rb') as f:
                    data = f.read()
                if self.logger:
                    self.logger.warning("Loading checkpoint without integrity validation")
                else:
                    print("Warning: Loading checkpoint without integrity validation")
            
            # Decompress and parse
            try:
                json_data = gzip.decompress(data).decode('utf-8')
            except gzip.BadGzipFile:
                # Handle old uncompressed checkpoints
                json_data = data.decode('utf-8')
            
            checkpoint_data = json.loads(json_data)
            
            # Validate configuration consistency (like mpii)
            if self.config_hash and checkpoint_data.get('config_hash'):
                if checkpoint_data['config_hash'] != self.config_hash:
                    if self.logger:
                        self.logger.warning("Configuration changed since checkpoint, starting fresh")
                    else:
                        print("Warning: Configuration changed since checkpoint, starting fresh")
                    return None
            
            # Restore state from checkpoint
            if 'processed_files' in checkpoint_data:
                self.processed_files = set(checkpoint_data['processed_files'])
            if 'failed_files' in checkpoint_data:
                self.failed_files = set(checkpoint_data['failed_files'])
            if 'processing_stats' in checkpoint_data:
                self.processing_stats.update(checkpoint_data['processing_stats'])
            if 'adaptive_interval' in checkpoint_data:
                self._adaptive_interval = checkpoint_data['adaptive_interval']
                
            return checkpoint_data
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to load checkpoint: {e}")
            else:
                print(f"Warning: Failed to load checkpoint: {e}")
            return None
    
    def initialize_model_results(self, model_name: str) -> None:
        """Initialize results files for a model - create empty files like mpii."""
        import json
        # Create empty JSON file for incremental results
        tmp_results_file = self.tmp_results_dir / f"{model_name}_results.json"
        with open(tmp_results_file, 'w') as f:
            json.dump([], f, indent=2)
        
        # Create CSV header for individual results
        results_file = self.individual_results_dir / f"{model_name}.csv"
        # Write just the header initially - results will be appended
        with open(results_file, 'w') as f:
            f.write("id,file_link\n")  # Will be updated with proper headers when first result is saved
    
    def track_file_processed(self, file_id: str, processing_time: float) -> None:
        """Track a processed file and update statistics like mpii."""
        self.processed_files.add(file_id)
        self._files_processed_since_last += 1
        
        # Update processing statistics
        self.processing_stats["files_completed"] += 1
        self.processing_stats["total_elapsed"] = time.time() - self._processing_start_time
        
        # Update average processing time
        if self.processing_stats["files_completed"] > 0:
            self.processing_stats["avg_time_per_file"] = (
                self.processing_stats["total_elapsed"] / self.processing_stats["files_completed"]
            )
    
    def track_file_failed(self, file_id: str, error: str = None) -> None:
        """Track a failed file like mpii."""
        self.failed_files.add(file_id)
        self.processing_stats["files_failed"] += 1
        if error:
            if self.logger:
                self.logger.warning(f"File {file_id} failed: {error}")
            else:
                print(f"Warning: File {file_id} failed: {error}")
    
    def save_single_result(self, result: VLMEvaluationResult) -> None:
        """Save a single result to tmp storage (CSV writing handled by intervals)."""
        import json
        
        model_name = result.model_name
        tmp_results_file = self.tmp_results_dir / f"{model_name}_results.json"
        
        # Load existing results from tmp file
        existing_results = []
        if tmp_results_file.exists():
            try:
                with open(tmp_results_file, 'r') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
        
        # Add new result to tmp file
        serializable_result = {
            "id": result.id,
            "file_link": result.file_link,
            "vcs_scores": result.vcs_scores,
            "model_name": result.model_name
        }
        existing_results.append(serializable_result)
        
        # Save updated tmp file
        with open(tmp_results_file, 'w') as f:
            json.dump(existing_results, f, indent=2)
    
    def save_interval_results_to_csv(self, model_name: str) -> None:
        """Save accumulated results to CSV during intervals like mpii."""
        tmp_results_file = self.tmp_results_dir / f"{model_name}_results.json"
        results_file = self.individual_results_dir / f"{model_name}.csv"
        
        if not tmp_results_file.exists():
            return
        
        # Load results from tmp file
        try:
            import json
            with open(tmp_results_file, 'r') as f:
                data = json.load(f)
            
            # Convert to VLMEvaluationResult objects
            results = []
            for item in data:
                result = VLMEvaluationResult(
                    id=item["id"],
                    file_link=item["file_link"],
                    vcs_scores=item["vcs_scores"],
                    model_name=item["model_name"]
                )
                results.append(result)
            
            # Save to CSV using ResultsProcessor
            ResultsProcessor.save_individual_results(
                results, 
                str(results_file),
                decimal_precision=3
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to save interval results for {model_name}: {e}")
            else:
                print(f"Warning: Failed to save interval results for {model_name}: {e}")
    
    def _append_result_to_csv(self, result: VLMEvaluationResult, csv_file_path):
        """Append a single result to CSV file with hierarchical MultiIndex columns."""
        import pandas as pd
        import os
        
        # Get unique chunk sizes and LCT values from the result
        chunk_sizes = set()
        lct_values = set()
        for vcs_key in result.vcs_scores.keys():
            if vcs_key.startswith('chunk'):
                chunk_part, lct_part = vcs_key.replace('chunk', '').split('_lct')
                chunk_sizes.add(int(chunk_part))
                lct_values.add(int(lct_part))
        
        # Sort for consistent ordering
        chunk_sizes = sorted(chunk_sizes)
        lct_values = sorted(lct_values)
        
        # Use pandas MultiIndex for proper CSV structure  
        self._append_result_to_csv_pandas(result, csv_file_path, chunk_sizes, lct_values)
    
    def _append_result_to_csv_pandas(self, result: VLMEvaluationResult, csv_file_path, chunk_sizes, lct_values):
        """Append a single result to CSV file with pandas MultiIndex columns."""
        import pandas as pd
        import os
        
        # Create hierarchical column structure with single spanning VCS header
        level_0 = ['id']
        level_1 = ['']
        level_2 = ['']
        
        # Build VCS columns with repeated VCS header
        for chunk_size in chunk_sizes:
            for lct in lct_values:
                # All VCS columns get 'VCS' header
                level_0.append('VCS')
                level_1.append(f'chunk_size={chunk_size}')
                level_2.append(f'LCT={lct}')
        
        # Add file_link column
        level_0.append('file_link')
        level_1.append('')
        level_2.append('')
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_arrays([level_0, level_1, level_2])
        
        # Create row data
        row_data = [result.id]
        
        # Add VCS scores in the same order as columns
        for chunk_size in chunk_sizes:
            for lct in lct_values:
                vcs_key = f"chunk{chunk_size}_lct{lct}"
                score = result.vcs_scores.get(vcs_key, 0.0)
                row_data.append(round(score, 3))
        
        # Add file_link
        row_data.append(result.file_link)
        
        # Create DataFrame with MultiIndex columns
        df_new = pd.DataFrame([row_data], columns=columns)
        
        # Check if file exists and has data
        file_exists = os.path.exists(csv_file_path)
        has_data = False
        
        if file_exists:
            try:
                # Try to read existing file to check if it has data
                df_existing = pd.read_csv(csv_file_path, header=[0,1,2])
                has_data = len(df_existing) > 0
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
                has_data = False
        
        if file_exists and has_data:
            # Append to existing file without headers
            df_new.to_csv(csv_file_path, mode='a', header=False, index=False)
        else:
            # Create new file with MultiIndex headers
            df_new.to_csv(csv_file_path, mode='w', header=True, index=False)

    def save_model_results(self, model_name: str, results: List[VLMEvaluationResult]) -> None:
        """Save individual model results to tmp folder and individual_results directory."""
        # Save to tmp checkpoint folder first (like mpii)
        import json
        tmp_results_file = self.tmp_results_dir / f"{model_name}_results.json"
        
        # Convert results to serializable format for tmp storage
        serializable_results = []
        for result in results:
            serializable_results.append({
                "id": result.id,
                "file_link": result.file_link,
                "vcs_scores": result.vcs_scores,
                "model_name": result.model_name
            })
        
        with open(tmp_results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Also save immediately to individual_results directory
        results_file = self.individual_results_dir / f"{model_name}.csv"
        ResultsProcessor.save_individual_results(
            results, 
            str(results_file),
            decimal_precision=3
        )
    
    def load_all_results(self) -> Dict[str, List[VLMEvaluationResult]]:
        """Load all saved model results from tmp checkpoint folder."""
        all_results = {}
        
        # Load from JSON files in tmp results directory
        for results_file in self.tmp_results_dir.glob("*_results.json"):
            model_name = results_file.stem.replace("_results", "")
            try:
                import json
                with open(results_file, 'r') as f:
                    data = json.load(f)
                
                results = []
                for item in data:
                    result = VLMEvaluationResult(
                        id=item["id"],
                        file_link=item["file_link"],
                        vcs_scores=item["vcs_scores"],
                        model_name=item["model_name"]
                    )
                    results.append(result)
                
                all_results[model_name] = results
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to load results for {model_name}: {e}")
                else:
                    print(f"Warning: Failed to load results for {model_name}: {e}")
        
        return all_results
    
    def clear_checkpoint(self) -> None:
        """Clear checkpoint and temporary files like mpii."""
        # Remove checkpoint file
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        
        # Remove checksum file
        if self.checksum_file.exists():
            self.checksum_file.unlink()
        
        # Clean up tmp results directory like mpii does on successful completion
        import shutil
        if self.tmp_results_dir.exists():
            shutil.rmtree(self.tmp_results_dir)
            if self.logger:
                self.logger.info(f"Cleaned up temporary results directory: {self.tmp_results_dir}")
            else:
                print(f"Cleaned up temporary results directory: {self.tmp_results_dir}")


class ResultsProcessor:
    """Handles processing and saving of evaluation results."""
    
    @staticmethod
    def save_individual_results(
        results: List[VLMEvaluationResult], 
        output_file: str,
        decimal_precision: int = 3
    ) -> None:
        """
        Save individual results to CSV file with hierarchical MultiIndex columns.
        
        Args:
            results: List of evaluation results
            output_file: Path to output CSV file
            decimal_precision: Number of decimal places for scores
        """
        if not results:
            # No need to log this as it's not critical
            return
        
        # Get all unique chunk sizes and LCT values from all results
        all_chunk_sizes = set()
        all_lct_values = set()
        for result in results:
            for vcs_key in result.vcs_scores.keys():
                if vcs_key.startswith('chunk'):
                    chunk_part, lct_part = vcs_key.replace('chunk', '').split('_lct')
                    all_chunk_sizes.add(int(chunk_part))
                    all_lct_values.add(int(lct_part))
        
        # Sort for consistent ordering
        chunk_sizes = sorted(all_chunk_sizes)
        lct_values = sorted(all_lct_values)
        
        # Create hierarchical column structure with single spanning VCS header
        level_0 = ['id']
        level_1 = ['']
        level_2 = ['']
        
        # Build VCS columns with repeated VCS header
        for chunk_size in chunk_sizes:
            for lct in lct_values:
                # All VCS columns get 'VCS' header
                level_0.append('VCS')
                level_1.append(f'chunk_size={chunk_size}')
                level_2.append(f'LCT={lct}')
        
        # Add file_link column
        level_0.append('file_link')
        level_1.append('')
        level_2.append('')
        
        # Create MultiIndex columns
        columns = pd.MultiIndex.from_arrays([level_0, level_1, level_2])
        
        # Create data rows
        data_rows = []
        for result in results:
            row_data = [result.id]
            
            # Add VCS scores in the same order as columns
            for chunk_size in chunk_sizes:
                for lct in lct_values:
                    vcs_key = f"chunk{chunk_size}_lct{lct}"
                    score = result.vcs_scores.get(vcs_key, 0.0)
                    row_data.append(round(score, decimal_precision))
            
            # Add file_link
            row_data.append(result.file_link)
            data_rows.append(row_data)
        
        # Create DataFrame with MultiIndex columns
        df = pd.DataFrame(data_rows, columns=columns)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with MultiIndex headers
        df.to_csv(output_file, index=False)
        # Don't print here - let the calling code handle logging
    
    @staticmethod
    def compute_aggregated_results(
        all_results: Dict[str, List[VLMEvaluationResult]],
        output_file: str,
        decimal_precision: int = 3
    ) -> None:
        """
        Compute and save aggregated results across all models.
        
        Args:
            all_results: Dictionary mapping model names to their results
            output_file: Path to output CSV file
            decimal_precision: Number of decimal places for scores
        """
        if not all_results:
            # No need to log this as it's not critical
            return
        
        aggregated_data = []
        
        for model_name, results in all_results.items():
            if not results:
                continue
            
            # Extract scores for each metric dynamically
            metrics = {}
            # Get all VCS score keys from first result
            if results:
                sample_result = results[0]
                for vcs_key in sample_result.vcs_scores.keys():
                    if vcs_key.startswith('chunk'):
                        chunk_part, lct_part = vcs_key.replace('chunk', '').split('_lct')
                        column_name = f"chunk_size={chunk_part}, LCT={lct_part}"
                        metrics[column_name] = [r.vcs_scores.get(vcs_key, 0.0) for r in results]
            
            # Compute mean ± std for each metric
            row = {'model_name': model_name}
            for metric_name, scores in metrics.items():
                scores_array = np.array(scores)
                mean_score = np.mean(scores_array)
                std_score = np.std(scores_array)
                row[metric_name] = f"{mean_score:.{decimal_precision}f} ± {std_score:.{decimal_precision}f}"
            
            aggregated_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(aggregated_data)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        # Don't print here - let the calling code handle logging


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
        global sat_adapted
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
    """Handles text embedding generation."""
    
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
        global model_nv, device_embed
        if model is None:
            model = model_nv
            
        if model is None:
            raise RuntimeError("NV-Embed model not initialized. Call ModelInitializer.initialize_nvembed() first.")
        
        device = device_embed if device_embed is not None else next(model.parameters()).device
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_np = model.encode(batch, instruction=instruction, max_length=max_length)
            embeddings = torch.tensor(embeddings_np, device=device, dtype=torch.float)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)


class ModelInitializer:
    """Handles initialization of deep learning models for VCS computation."""
    
    # Class-level storage for models
    _sat_model = None
    _tokenizer_nv = None
    _model_nv = None
    _device_embed = None
    
    @classmethod
    def initialize_sat(cls, sat_model_name: str = "sat-12l-sm") -> None:
        """Initialize SAT model for sentence segmentation."""
        global sat_adapted
        
        cls._sat_model = SaT(sat_model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._sat_model.half().to(device)
        
        # Set the global variable in this module
        sat_adapted = cls._sat_model

    @classmethod
    def initialize_nvembed(cls, config: Dict) -> None:
        """Initialize NV-embed model for semantic embeddings."""
        global model_nv, device_embed
        
        model_path = Path(config['models']['nv_embed_path'])
        use_cuda = config.get('advanced', {}).get('use_cuda', True)
        cls._device_embed = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        cls._tokenizer_nv = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        cls._model_nv = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        cls._tokenizer_nv.padding_side = "right"
        cls._model_nv.eval()
        cls._model_nv.to(cls._device_embed)
        
        # Set the global variables in this module
        model_nv = cls._model_nv
        device_embed = cls._device_embed
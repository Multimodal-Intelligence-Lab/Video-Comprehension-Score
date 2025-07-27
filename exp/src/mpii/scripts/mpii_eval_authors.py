"""
Comprehensive Evaluation Framework for Authors Experiments

This module provides a unified evaluation framework for authors experiments,
handling cross-author comparisons following research and industry best practices.

Metrics included:
- Traditional N-gram based: BLEU-1, BLEU-4
- Semantic similarity: METEOR, ROUGE-1, ROUGE-4, ROUGE-L, ROUGE-Lsum
- Novel semantic coherence: VCS (Video Comprehension Score)

Enhanced Features:
- Structured JSON logging for analysis
- Parallel processing for efficiency
- Checkpointing for resuming interrupted evaluations
- Cross-author comparison matrix generation

Authors: Research Team
"""

import torch
import torch.nn.functional as F
import re
import string
import os
import json
import glob
import pandas as pd
import numpy as np
import time
import pickle
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel
import contractions
from wtpsplit import SaT
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import vcs
from utils.traditional_metrics import evaluate_BRM
from utils.core import (
    ConfigLoader, CheckpointManager, ModelInitializer, 
    TextProcessor, EmbeddingGenerator,
    PUNCTUATIONS, DECIMAL_PRECISION
)

# Global model instances
sat_adapted = None
tokenizer_nv = None
model_nv = None
device_embed = None


# ConfigLoader now imported from utils.core

# Override _validate_config for authors-specific validation
class AuthorsConfigLoader(ConfigLoader):
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate required configuration fields for authors experiments."""
        # Call parent validation first
        ConfigLoader._validate_config(config)
        
        # Authors-specific validation
        required_sections = ['authors']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate authors specific settings
        if 'author_files' not in config['authors']:
            raise ValueError("Missing required field: authors.author_files")
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration with base merging and authors-specific validation."""
        # Use parent class method for base config merging
        config = ConfigLoader.load_config(config_path)
        # Add authors-specific validation
        AuthorsConfigLoader._validate_config(config)
        return config


@dataclass
class ComparisonResult:
    """Container for evaluation metrics from a single comparison."""
    ref_author: str
    other_author: str
    index_str: str
    metrics: Dict[str, float]


@dataclass
class AggregatedResults:
    """Container for aggregated evaluation results across multiple comparisons."""
    means: Dict[str, float]
    stds: Dict[str, float]
    n_samples: int


class StructuredLogger:
    """Enhanced logging system with JSON format for automated analysis."""
    
    def __init__(self, log_dir: str, experiment_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup loggers
        self.setup_loggers()
        
    def setup_loggers(self):
        """Setup structured logging with JSON format."""
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # JSON structured logger
        self.json_logger = logging.getLogger(f"json_{self.experiment_id}")
        self.json_logger.setLevel(logging.INFO)
        self.json_logger.handlers.clear()
        
        # Main execution logger
        self.main_logger = logging.getLogger(f"main_{self.experiment_id}")
        self.main_logger.setLevel(logging.INFO)
        self.main_logger.handlers.clear()
        
        main_handler = logging.FileHandler(self.log_dir / f"mpii_authors_{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    def log_author_start(self, ref_author: str, total_comparisons: int):
        """Log start of author processing."""
        self.main_logger.info(f"Processing reference author: {ref_author} ({total_comparisons} comparisons)")
    
    def log_author_complete(self, ref_author: str, processing_time: float, comparisons_processed: int):
        """Log completion of author processing."""
        self.main_logger.info(f"Completed reference author: {ref_author} ({comparisons_processed} comparisons, {processing_time:.2f}s)")
    
    def log_comparison(self, ref_author: str, other_author: str, index_str: str, 
                      processing_time: float, metrics: Dict[str, float]):
        """Log individual comparison metrics."""
        # Simplified logging - only log errors for comparisons
        pass
    
    def log_progress(self, current: int, total: int, description: str = "Processing"):
        """Log progress information."""
        self.main_logger.info(f"{description}: {current}/{total} ({current/total*100:.1f}%)")
        
        if current > 0:
            elapsed = time.time() - getattr(self, '_start_time', time.time())
            eta = (elapsed / current) * (total - current)
            self.main_logger.info(f"ETA: {eta/60:.1f} minutes")
    
    def log_error(self, message: str, error: Exception, context: Dict = None):
        """Log errors with context."""
        self.main_logger.error(f"{message}: {error}")
    
    def set_start_time(self):
        """Set the start time for progress tracking."""
        self._start_time = time.time()


# CheckpointManager now imported from utils.core


class ParallelProcessor:
    """Handles parallel processing of author comparisons."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
    
    def process_comparisons_parallel(self, comparison_tasks: List[Tuple], 
                                    evaluator: 'MetricsEvaluator',
                                    logger: StructuredLogger,
                                    checkpoint_manager: CheckpointManager,
                                    config: Dict = None) -> None:
        """Process individual comparisons in parallel (like comparison script)."""
        processed_comparisons = set()
        failed_comparisons = set()
        comparisons_since_last_checkpoint = 0
        start_time = time.time()
        
        # Load checkpoint with configuration validation
        checkpoint_state = checkpoint_manager.load_checkpoint(config)
        if checkpoint_state:
            processed_comparisons = set(checkpoint_state.get("processed_files", []))
            failed_comparisons = set(checkpoint_state.get("failed_files", []))
            logger.main_logger.info(f"Resumed from checkpoint: {len(processed_comparisons)} comparisons processed, {len(failed_comparisons)} failed")
        
        # Filter remaining tasks (exclude both processed and permanently failed)
        remaining_tasks = [task for task in comparison_tasks 
                          if task[0] not in processed_comparisons and task[0] not in failed_comparisons]
        
        if not remaining_tasks:
            logger.main_logger.info("All comparisons already processed")
            return
        
        logger.main_logger.info(f"Processing {len(remaining_tasks)} comparisons with {self.max_workers} workers")
        logger.set_start_time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in remaining_tasks:
                future = executor.submit(evaluator.process_single_comparison, *task)
                future_to_task[future] = task
            
            # Process results as they complete
            completed = len(processed_comparisons)
            total = len(comparison_tasks)
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                task_id = task[0]
                
                try:
                    comparison_result = future.result()
                    completed += 1
                    comparisons_since_last_checkpoint += 1
                    
                    # Calculate timing
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    
                    if comparison_result:
                        processed_comparisons.add(task_id)
                        # Save individual comparison result for memory efficiency
                        checkpoint_manager.save_file_results(task_id, [comparison_result])
                    else:
                        logger.main_logger.warning(f"No result from comparison: {task_id}")
                        failed_comparisons.add(task_id)
                    
                    logger.log_progress(completed, total, "Comparisons")
                    
                    # Adaptive checkpoint saving (every N comparisons)
                    processing_stats = {
                        "avg_time_per_file": elapsed_time / max(1, completed),
                        "total_elapsed": elapsed_time,
                        "files_completed": completed
                    }
                    
                    if checkpoint_manager.should_save_checkpoint(comparisons_since_last_checkpoint):
                        checkpoint_manager.save_checkpoint(
                            processed_files=list(processed_comparisons),
                            failed_files=list(failed_comparisons),
                            processing_stats=processing_stats
                        )
                        comparisons_since_last_checkpoint = 0
                        interval = checkpoint_manager.get_adaptive_interval()
                        logger.main_logger.info(f"Checkpoint saved at {completed}/{total} comparisons (interval: {interval})")
                
                except Exception as e:
                    logger.log_error(f"Failed to process comparison: {task_id}", e, {"task_id": task_id})
                    failed_comparisons.add(task_id)
                    completed += 1
                    comparisons_since_last_checkpoint += 1
        
        # Final checkpoint with complete statistics
        final_time = time.time()
        final_stats = {
            "avg_time_per_file": (final_time - start_time) / max(1, len(remaining_tasks)),
            "total_elapsed": final_time - start_time,
            "files_completed": completed,
            "success_rate": len(processed_comparisons) / max(1, len(processed_comparisons) + len(failed_comparisons))
        }
        
        checkpoint_manager.save_checkpoint(
            processed_files=list(processed_comparisons),
            failed_files=list(failed_comparisons),
            processing_stats=final_stats
        )


# ModelInitializer now imported from utils.core


# TextProcessor now imported from utils.core


# EmbeddingGenerator now imported from utils.core


class MetricsEvaluator:
    """Core evaluation engine for computing all metrics with logging."""
    
    def __init__(self, config: Dict, logger: StructuredLogger = None):
        self.config = config
        self.segmenter = TextProcessor.sat_segmenter
        self.embedding_fn = EmbeddingGenerator.nv_embed_embedding_fn
        self.logger = logger
        self.selected_metrics = config['authors']['selected_metrics']
    
    def _generate_zero_vcs_metrics(self) -> Dict[str, float]:
        """Generate zero VCS metrics based on config."""
        vcs_config = self.config['vcs']
        chunk_sizes = vcs_config.get('chunk_size', [1])
        if not isinstance(chunk_sizes, list):
            chunk_sizes = [chunk_sizes]  # Convert single value to list
        lct_values = vcs_config['lct']
        
        zero_vcs_metrics = {}
        for chunk_size in chunk_sizes:
            for lct in lct_values:
                zero_vcs_metrics[f"VCS_C{chunk_size}_LCT{lct}"] = 0.0
        
        return zero_vcs_metrics
    
    def process_single_comparison(self, task_id: str, ref_author: str, other_author: str, 
                                 index_str: str, authors_data: Dict, output_folder: str) -> ComparisonResult:
        """Process a single comparison and return result."""
        try:
            ref_text = authors_data[ref_author][index_str]
            other_text = authors_data[other_author][index_str]
            
            result = self.evaluate_single_comparison(
                reference=ref_text,
                generated=other_text,
                ref_author=ref_author,
                other_author=other_author,
                index_str=index_str
            )
            
            # Write individual result immediately (like comparison script)
            if result:
                self._save_individual_comparison_result(result, output_folder)
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error processing comparison: {task_id}", e, {"task_id": task_id})
            return None
    
    def process_single_author(self, ref_author: str, authors_data: Dict, output_folder: str) -> List[ComparisonResult]:
        """Process a single reference author and return results."""
        start_time = time.time()
        author_results = []
        
        try:
            ref_data = authors_data[ref_author]
            other_authors = [author for author in authors_data.keys() if author != ref_author]
            
            total_comparisons = len(ref_data) * len(other_authors)
            
            if self.logger:
                self.logger.log_author_start(ref_author, total_comparisons)
            
            # Process each index of the reference author
            for index_str, ref_text in ref_data.items():
                # Compare against all other authors' same index
                for other_author in other_authors:
                    other_data = authors_data[other_author]
                    
                    # Only compare if the other author has the same index
                    if index_str in other_data:
                        other_text = other_data[index_str]
                        
                        result = self.evaluate_single_comparison(
                            reference=ref_text,
                            generated=other_text,
                            ref_author=ref_author,
                            other_author=other_author,
                            index_str=index_str
                        )
                        author_results.append(result)
            
            # Save individual author results
            if author_results:
                individual_results_dir = self.config['paths'].get('individual_results_dir', 'individual_results')
                individual_output_folder = Path(output_folder) / individual_results_dir / ref_author
                individual_output_folder.mkdir(parents=True, exist_ok=True)
                self._save_author_results(author_results, ref_author, individual_output_folder)
            
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.log_author_complete(ref_author, processing_time, len(author_results))
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error processing author: {ref_author}", e, {"ref_author": ref_author})
        
        return author_results
    
    def evaluate_single_comparison(
        self, 
        reference: str, 
        generated: str, 
        ref_author: str, 
        other_author: str,
        index_str: str
    ) -> ComparisonResult:
        """
        Evaluate a single comparison against reference with timing.
        
        Args:
            reference: Reference text (from ref_author)
            generated: Generated text (from other_author)
            ref_author: Reference author identifier
            other_author: Other author identifier
            index_str: Index identifier
            
        Returns:
            ComparisonResult containing all computed metrics
        """
        start_time = time.time()
        
        try:
            # Compute traditional metrics (BLEU, METEOR, ROUGE)
            brm_results = evaluate_BRM(reference, generated)
            
            # Extract VCS configuration
            vcs_config = self.config['vcs']
            lct_values = vcs_config['lct']
            chunk_sizes = vcs_config.get('chunk_size', [1])
            if not isinstance(chunk_sizes, list):
                chunk_sizes = [chunk_sizes]  # Convert single value to list for compatibility
            context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
            context_window = vcs_config.get('context_window_control', 4.0)
            
            # Compute VCS metrics for each chunk_size × LCT combination
            vcs_metrics = {}
            for chunk_size in chunk_sizes:
                for lct in lct_values:
                    try:
                        vcs_results = vcs.compute_vcs_score(
                            reference_text=reference,
                            generated_text=generated,
                            segmenter_fn=self.segmenter,
                            embedding_fn_las=self.embedding_fn,
                            embedding_fn_gas=self.embedding_fn,
                            chunk_size=chunk_size,
                            context_cutoff_value=context_cutoff,
                            context_window_control=context_window,
                            lct=lct,
                            return_all_metrics=True,
                            return_internals=False
                        )
                        vcs_metrics[f"VCS_C{chunk_size}_LCT{lct}"] = vcs_results.get("VCS", 0.0)
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(f"VCS computation failed for chunk_size={chunk_size}, LCT={lct}", e)
                        vcs_metrics[f"VCS_C{chunk_size}_LCT{lct}"] = 0.0
            
            # Extract and organize metrics
            metrics = {
                "BLEU-1": brm_results.get("bleu1", 0.0),
                "BLEU-4": brm_results.get("bleu4", 0.0),
                "METEOR": brm_results.get("meteor", 0.0),
                "ROUGE-1": brm_results.get("rouge", {}).get("rouge1", 0.0),
                "ROUGE-4": brm_results.get("rouge", {}).get("rouge4", 0.0),
                "ROUGE-L": brm_results.get("rouge", {}).get("rougeL", 0.0),
                "ROUGE-Lsum": brm_results.get("rouge", {}).get("rougeLsum", 0.0),
                **vcs_metrics
            }
            
            # Log comparison metrics
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.log_comparison(ref_author, other_author, index_str, 
                                         processing_time, metrics)
            
            return ComparisonResult(
                ref_author=ref_author,
                other_author=other_author,
                index_str=index_str,
                metrics=metrics
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error evaluating comparison: {ref_author} vs {other_author}", e, 
                                    {"ref_author": ref_author, "other_author": other_author, "index_str": index_str})
            
            # Return zero metrics on failure
            zero_metrics = {
                "BLEU-1": 0.0,
                "BLEU-4": 0.0,
                "METEOR": 0.0,
                "ROUGE-1": 0.0,
                "ROUGE-4": 0.0,
                "ROUGE-L": 0.0,
                "ROUGE-Lsum": 0.0,
                **self._generate_zero_vcs_metrics()  # Dynamic VCS metrics based on config
            }
            return ComparisonResult(
                ref_author=ref_author,
                other_author=other_author,
                index_str=index_str,
                metrics=zero_metrics
            )
    
    def _save_individual_comparison_result(self, result: ComparisonResult, output_folder: str):
        """Save individual comparison result immediately to CSV (like comparison script)."""
        import pandas as pd
        import threading
        
        # Thread-safe file writing
        if not hasattr(self, '_write_lock'):
            self._write_lock = threading.Lock()
        
        with self._write_lock:
            try:
                # Create directory structure
                individual_results_dir = self.config['paths'].get('individual_results_dir', 'individual_results')
                individual_output_folder = Path(output_folder) / individual_results_dir / result.ref_author
                individual_output_folder.mkdir(parents=True, exist_ok=True)
                
                # File path for this specific index
                csv_path = individual_output_folder / f"{result.ref_author}_{result.index_str}.csv"
                
                # Get decimal precision from config
                decimal_precision = self.config.get('output', {}).get('decimal_precision', 3)
                
                # Prepare row data
                row_data = {"Comparison": f"{result.ref_author}-{result.other_author}"}
                for metric, value in result.metrics.items():
                    row_data[metric] = round(value, decimal_precision)
                
                # Check if file exists to determine if we need to append
                if csv_path.exists():
                    # Append to existing file
                    df_existing = pd.read_csv(csv_path)
                    df_new = pd.DataFrame([row_data])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_path, index=False)
                else:
                    # Create new file
                    df = pd.DataFrame([row_data])
                    df.to_csv(csv_path, index=False)
                    
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Failed to save individual result", e, 
                                        {"ref_author": result.ref_author, "index_str": result.index_str})
    
    def _save_author_results(self, results: List[ComparisonResult], ref_author: str, output_dir: Path):
        """Save individual author results to CSV with separate files for each index."""
        if not results:
            return
            
        # Get decimal precision from config
        decimal_precision = self.config.get('output', {}).get('decimal_precision', 3)
        
        # Get all metric names (including VCS_C{chunk_size}_LCT{lct} variants)
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Sort metrics for consistent ordering
        traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
        vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
        ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
        
        # Group results by index_str
        results_by_index = {}
        for result in results:
            index_str = result.index_str
            if index_str not in results_by_index:
                results_by_index[index_str] = []
            results_by_index[index_str].append(result)
        
        # Create separate CSV file for each index
        for index_str, index_results in results_by_index.items():
            # Build data with comparisons as rows and metrics as columns
            data = []
            for result in index_results:
                row = {"Comparison": f"{result.ref_author}-{result.other_author}"}
                for metric in ordered_metrics:
                    row[metric] = round(result.metrics.get(metric, 0.0), decimal_precision)
                data.append(row)
            
            df = pd.DataFrame(data)
            csv_path = output_dir / f"{ref_author}_{index_str}.csv"
            df.to_csv(csv_path, index=False)


class DataLoader:
    """Handles loading and parsing of evaluation datasets."""
    
    @staticmethod
    def load_author_files(data_dir: str, author_files: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """Load all author JSON files."""
        authors_data = {}
        
        for author_id, filename in author_files.items():
            file_path = Path(data_dir) / filename
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                authors_data[author_id] = data
            except Exception as e:
                raise FileNotFoundError(f"Error loading {file_path}: {e}")
        
        return authors_data


class ResultsManager:
    """Handles saving and aggregating evaluation results."""
    
    @staticmethod
    def aggregate_results(all_results: List[List[ComparisonResult]]) -> Dict[str, AggregatedResults]:
        """
        Aggregate results by reference author.
        
        Args:
            all_results: List of result lists from each author
            
        Returns:
            Dictionary mapping author IDs to AggregatedResults
        """
        # Flatten all results
        flattened = [result for author_results in all_results for result in author_results]
        
        # Group by reference author and other author
        grouped = {}
        for result in flattened:
            ref_key = result.ref_author
            other_key = result.other_author
            
            if ref_key not in grouped:
                grouped[ref_key] = {}
            if other_key not in grouped[ref_key]:
                grouped[ref_key][other_key] = []
            
            grouped[ref_key][other_key].append(result.metrics)
        
        # Compute statistics for each reference author
        aggregated_by_author = {}
        
        for ref_author, other_authors_data in grouped.items():
            author_means = {}
            author_stds = {}
            total_samples = 0
            
            for other_author, metrics_list in other_authors_data.items():
                # Convert to DataFrame for easier computation
                df = pd.DataFrame(metrics_list)
                means = df.mean().to_dict()
                stds = df.std().to_dict()
                
                # Store with comparison key
                comp_key = f"{ref_author}-{other_author}"
                author_means[comp_key] = means
                author_stds[comp_key] = stds
                total_samples += len(metrics_list)
            
            aggregated_by_author[ref_author] = AggregatedResults(
                means=author_means,
                stds=author_stds,
                n_samples=total_samples
            )
        
        return aggregated_by_author
    
    @staticmethod
    def save_aggregated_results(
        aggregated_by_author: Dict[str, AggregatedResults], 
        output_dir: str,
        config: Dict = None
    ) -> None:
        """Save aggregated results for each author."""
        os.makedirs(output_dir, exist_ok=True)
        
        decimal_precision = config.get('output', {}).get('decimal_precision', 3) if config else 3
        
        # Save individual author aggregated results
        for ref_author, aggregated in aggregated_by_author.items():
            # Get all comparisons and metrics
            comparisons = list(aggregated.means.keys())
            comparisons.sort()  # Sort for consistent ordering
            
            # Get all metrics from the first comparison
            all_metrics = set()
            for means in aggregated.means.values():
                all_metrics.update(means.keys())
            
            # Sort metrics for consistent ordering
            traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
            vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
            ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
            
            # Build data with comparisons as rows and metrics as columns
            data = []
            for comparison in comparisons:
                row = {"Comparison": comparison}
                means = aggregated.means[comparison]
                stds = aggregated.stds[comparison]
                
                # Format as "mean ± std" for each metric
                for metric in ordered_metrics:
                    row[metric] = f"{means.get(metric, 0.0):.{decimal_precision}f} ± {stds.get(metric, 0.0):.{decimal_precision}f}"
                
                data.append(row)
            
            df = pd.DataFrame(data)
            output_path = Path(output_dir) / f"{ref_author}.csv"
            df.to_csv(output_path, index=False)
        
        # Create combined aggregated results
        ResultsManager._save_combined_results(aggregated_by_author, output_dir, decimal_precision)
    
    @staticmethod
    def _save_combined_results(aggregated_by_author: Dict[str, AggregatedResults], 
                             output_dir: str, decimal_precision: int):
        """Save combined results across all authors."""
        
        # Collect all comparisons from all authors
        all_data = []
        
        for ref_author, aggregated in aggregated_by_author.items():
            # Get all metrics from the first comparison
            all_metrics = set()
            for means in aggregated.means.values():
                all_metrics.update(means.keys())
            
            # Sort metrics for consistent ordering
            traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
            vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
            ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
            
            for comparison, means in aggregated.means.items():
                stds = aggregated.stds[comparison]
                row = {"Comparison": comparison}
                
                # Format as "mean ± std" for each metric
                for metric in ordered_metrics:
                    row[metric] = f"{means.get(metric, 0.0):.{decimal_precision}f} ± {stds.get(metric, 0.0):.{decimal_precision}f}"
                
                all_data.append(row)
        
        # Create combined DataFrame
        df_combined = pd.DataFrame(all_data)
        
        # Sort by comparison for consistent ordering
        df_combined = df_combined.sort_values('Comparison').reset_index(drop=True)
        
        # Save combined results
        combined_path = Path(output_dir) / "aggr_comp_authors.csv"
        df_combined.to_csv(combined_path, index=False)


class EvaluationPipeline:
    """Enhanced evaluation pipeline with logging, parallel processing, and checkpointing."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Extract paths from config
        paths = config['paths']
        
        # Handle relative paths - make them relative to the script location
        script_dir = Path(__file__).parent.parent
        
        # Input folder
        data_dir = Path(paths['data_dir'])
        self.input_folder = str(data_dir if data_dir.is_absolute() else script_dir / data_dir)
        
        # Output folder
        results_dir = Path(paths['results_dir'])
        self.output_folder = str(results_dir if results_dir.is_absolute() else script_dir / results_dir)
        
        # Logs directory  
        logs_dir_path = Path(paths.get('logs_dir', 'logs'))
        logs_dir = str(logs_dir_path if logs_dir_path.is_absolute() else script_dir / logs_dir_path)
        # Create authors-specific logs directory
        authors_logs_dir = Path(logs_dir) / 'authors'
        
        # Extract processing settings
        processing = config['processing']
        max_workers = processing.get('max_workers', 4)
        self.resume_from_checkpoint = processing.get('resume_from_checkpoint', True)
        
        # Extract authors settings
        self.author_files = config['authors']['author_files']
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(authors_logs_dir, exist_ok=True)
        
        # Intelligent experiment ID handling
        experiment_config = config.get('experiment', {})
        config_experiment_id = experiment_config.get('experiment_id')
        
        if not config_experiment_id or config_experiment_id.strip() == "":
            # Case 1: No experiment ID in config → start fresh with new ID
            self.experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Starting fresh evaluation with new experiment ID: {self.experiment_id}")
        else:
            # Case 2: Experiment ID provided → check for checkpoint files
            config_experiment_id = config_experiment_id.strip()
            checkpoint_file = Path(self.output_folder) / f"checkpoint_{config_experiment_id}.json.gz"
            
            if checkpoint_file.exists():
                # Resume from existing checkpoint + use existing log file
                self.experiment_id = config_experiment_id
                print(f"Found checkpoint for experiment ID: {self.experiment_id}, will attempt to resume")
            else:
                # No checkpoint found → start fresh with new ID + new log file
                self.experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"No checkpoint found for experiment ID: {config_experiment_id}")
                print(f"Starting fresh evaluation with new experiment ID: {self.experiment_id}")
        
        # Generate configuration hash for consistency validation
        import hashlib
        import json
        config_copy = config.copy()
        config_copy.pop('experiment', None)  # Remove experiment-specific fields
        config_str = json.dumps(config_copy, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Initialize enhanced components
        self.logger = StructuredLogger(str(authors_logs_dir), self.experiment_id)
        self.checkpoint_manager = CheckpointManager(self.output_folder, self.experiment_id, config_hash)
        self.parallel_processor = ParallelProcessor(max_workers)
        self.evaluator = MetricsEvaluator(config, self.logger)
        self.results_manager = ResultsManager()
        
        # Initialize models
        self.logger.main_logger.info("Initializing models...")
        ModelInitializer.initialize_sat()
        ModelInitializer.initialize_nvembed(config)
        self.logger.main_logger.info("Models initialized successfully")
    
    def run_evaluation(self) -> None:
        """Execute the complete evaluation pipeline with enhancements."""
        
        self.logger.main_logger.info(f"Starting authors evaluation pipeline")
        self.logger.main_logger.info(f"Input folder: {self.input_folder}")
        self.logger.main_logger.info(f"Output folder: {self.output_folder}")
        self.logger.main_logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.main_logger.info(f"Authors: {list(self.author_files.keys())}")
        
        start_time = time.time()
        
        try:
            # Load all author data
            authors_data = DataLoader.load_author_files(self.input_folder, self.author_files)
            
            self.logger.main_logger.info(f"Loaded {len(authors_data)} authors")
            for author_id, data in authors_data.items():
                self.logger.main_logger.info(f"  {author_id}: {len(data)} entries")
            
            # Create tasks for each individual comparison (like comparison script)
            comparison_tasks = []
            for ref_author in authors_data.keys():
                ref_data = authors_data[ref_author]
                other_authors = [author for author in authors_data.keys() if author != ref_author]
                
                # Create task for each comparison combination
                for index_str in ref_data.keys():
                    for other_author in other_authors:
                        if index_str in authors_data[other_author]:
                            task_id = f"{ref_author}_{other_author}_{index_str}"
                            comparison_tasks.append((task_id, ref_author, other_author, index_str, authors_data, self.output_folder))
            
            self.logger.main_logger.info(f"Total comparisons to process: {len(comparison_tasks)}")
            
            # Process comparisons in parallel with checkpointing (like comparison script)
            self.parallel_processor.process_comparisons_parallel(
                comparison_tasks, self.evaluator, self.logger, self.checkpoint_manager, self.config
            )
            
            # Load all results from checkpoint files for aggregation
            all_results = self.checkpoint_manager.load_all_results()
            
            # Aggregate and save final results
            if all_results:
                self.logger.main_logger.info("Aggregating results...")
                aggregated_by_author = self.results_manager.aggregate_results(all_results)
                
                aggregated_results_dir = self.config['paths'].get('aggregated_results_dir', 'aggregated_results')
                aggregated_output_folder = Path(self.output_folder) / aggregated_results_dir
                aggregated_output_folder.mkdir(parents=True, exist_ok=True)
                
                self.results_manager.save_aggregated_results(
                    aggregated_by_author, str(aggregated_output_folder), self.config
                )
                
                # Calculate statistics
                total_comparisons = sum(len(author_results) for author_results in all_results)
                total_time = time.time() - start_time
                
                self.logger.main_logger.info(f"Authors evaluation completed successfully!")
                self.logger.main_logger.info(f"Total comparisons processed: {total_comparisons}")
                self.logger.main_logger.info(f"Total processing time: {total_time:.2f} seconds")
                if total_comparisons > 0:
                    self.logger.main_logger.info(f"Average time per comparison: {total_time/total_comparisons:.3f} seconds")
                else:
                    self.logger.main_logger.info("Average time per comparison: N/A (no comparisons processed)")
                
                # Log completion summary
                self.logger.main_logger.info(f"Results saved to: {aggregated_output_folder}")
                
                # Clean up checkpoint
                self.checkpoint_manager.clear_checkpoint()
                self.logger.main_logger.info("Checkpoint cleaned up")
            else:
                self.logger.main_logger.warning("No results generated")
        
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VCS Authors Evaluation Framework - Comprehensive cross-author comparison evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mpii_eval_authors.py --config config.yaml
  python mpii_eval_authors.py --config config.yaml --verbose

For detailed configuration options, see config.yaml file.
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output (overrides config setting)'
    )
    
    parser.add_argument(
        '--experiment_id', 
        type=str,
        help='Override experiment ID from config'
    )
    
    return parser.parse_args()


def main():
    """Enhanced main execution function with config-based configuration."""
    args = parse_arguments()
    
    try:
        # Load configuration
        config = AuthorsConfigLoader.load_config(args.config)
        
        # Override experiment ID if provided
        if args.experiment_id:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['experiment_id'] = args.experiment_id
        
        # Override verbosity if specified
        if args.verbose:
            config.setdefault('logging', {})['verbose'] = True
        
        # Initialize and run pipeline
        pipeline = EvaluationPipeline(config)
        pipeline.run_evaluation()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
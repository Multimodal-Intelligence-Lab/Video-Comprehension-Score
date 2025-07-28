"""
Comprehensive Evaluation Framework for Text Generation Models

This module provides a standardized evaluation framework following research and industry 
best practices for evaluating text generation models using multiple complementary metrics.

Metrics included:
- Traditional N-gram based: BLEU-1, BLEU-4
- Semantic similarity: METEOR, ROUGE-1, ROUGE-4, ROUGE-L, ROUGE-Lsum
- Novel semantic coherence: VCS (Video Comprehension Score)

Enhanced Features:
- Structured JSON logging for analysis
- Parallel processing for efficiency
- Checkpointing for resuming interrupted evaluations

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
    PUNCTUATIONS, DECIMAL_PRECISION, COMPARISON_METRICS_ORDER
)

# Global model instances
sat_adapted = None
tokenizer_nv = None
model_nv = None
device_embed = None

# Constants
METRICS_ORDER = COMPARISON_METRICS_ORDER


# ConfigLoader now imported from utils.core
# No experiment-specific validation needed for comparison


@dataclass
class EvaluationResult:
    """Container for evaluation metrics from a single test case."""
    test_case_id: str
    test_case_name: str
    metrics: Dict[str, float]


@dataclass
class AggregatedResults:
    """Container for aggregated evaluation results across multiple runs."""
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
        
        # Simplified logging - no verbose JSON files
        
        # Main execution logger
        self.main_logger = logging.getLogger(f"main_{self.experiment_id}")
        self.main_logger.setLevel(logging.INFO)
        self.main_logger.handlers.clear()
        
        main_handler = logging.FileHandler(self.log_dir / f"mpii_comparison_{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    class JSONFormatter(logging.Formatter):
        """Custom JSON formatter for structured logging."""
        
        def format(self, record):
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName
            }
            
            # Add any extra fields
            for key, value in record.__dict__.items():
                if key.startswith(('metric_', 'timing_', 'resource_', 'file_')):
                    log_entry[key] = value
            
            return json.dumps(log_entry)
    
    def log_file_start(self, file_path: str, total_test_cases: int):
        """Log start of file processing."""
        self.main_logger.info(f"Processing file: {file_path} ({total_test_cases} test cases)")
    
    def log_file_complete(self, file_path: str, processing_time: float, test_cases_processed: int):
        """Log completion of file processing."""
        self.main_logger.info(f"Completed file: {file_path} ({test_cases_processed} test cases, {processing_time:.2f}s)")
    
    def log_test_case(self, test_case_id: str, test_case_name: str, 
                     file_path: str, processing_time: float, metrics: Dict[str, float]):
        """Log individual test case metrics."""
        # Simplified logging - only log errors for test cases
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
    """Handles parallel processing of files."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
    
    def process_files_parallel(self, file_tasks: List[Tuple], 
                             evaluator: 'MetricsEvaluator',
                             logger: StructuredLogger,
                             checkpoint_manager: CheckpointManager,
                             config: Dict = None) -> List[List[EvaluationResult]]:
        """Process files in parallel with enhanced checkpointing."""
        
        all_results = []
        processed_files = set()
        failed_files = set()
        files_since_last_checkpoint = 0
        start_time = time.time()
        
        # Load checkpoint with configuration validation
        checkpoint_state = checkpoint_manager.load_checkpoint(config)
        if checkpoint_state:
            processed_files = set(checkpoint_state.get("processed_files", []))
            failed_files = set(checkpoint_state.get("failed_files", []))
            all_results = checkpoint_manager.load_all_results()  # Load from separate files
            logger.main_logger.info(f"Resumed from checkpoint: {len(processed_files)} files processed, {len(failed_files)} failed")
        
        # Filter remaining files (exclude both processed and permanently failed)
        remaining_tasks = [task for task in file_tasks 
                          if task[0] not in processed_files and task[0] not in failed_files]
        
        if not remaining_tasks:
            logger.main_logger.info("All files already processed")
            return all_results
        
        logger.main_logger.info(f"Processing {len(remaining_tasks)} files with {self.max_workers} workers")
        logger.set_start_time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in remaining_tasks:
                future = executor.submit(evaluator.process_single_file, *task)
                future_to_task[future] = task
            
            # Process results as they complete
            completed = len(processed_files)
            total = len(file_tasks)
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                file_path = task[0]
                
                try:
                    file_results = future.result()
                    if file_results:
                        all_results.append(file_results)
                        processed_files.add(file_path)
                        # Save individual file results for memory efficiency
                        checkpoint_manager.save_file_results(file_path, file_results)
                    else:
                        logger.main_logger.warning(f"No results from file: {file_path}")
                    
                    completed += 1
                    files_since_last_checkpoint += 1
                    logger.log_progress(completed, total, "Files")
                    
                    # Adaptive checkpoint saving
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    avg_time_per_file = elapsed_time / max(1, completed - len(checkpoint_state.get("processed_files", [])) if checkpoint_state else completed)
                    
                    processing_stats = {
                        "avg_time_per_file": avg_time_per_file,
                        "total_elapsed": elapsed_time,
                        "files_completed": completed
                    }
                    
                    if checkpoint_manager.should_save_checkpoint(files_since_last_checkpoint):
                        checkpoint_manager.save_checkpoint(
                            processed_files=list(processed_files),
                            failed_files=list(failed_files),
                            processing_stats=processing_stats
                        )
                        files_since_last_checkpoint = 0
                        interval = checkpoint_manager.get_adaptive_interval()
                        logger.main_logger.info(f"Checkpoint saved at {completed}/{total} files (interval: {interval})")
                
                except Exception as e:
                    logger.log_error(f"Failed to process file: {file_path}", e, {"file_path": file_path})
                    failed_files.add(file_path)
                    completed += 1
                    files_since_last_checkpoint += 1
        
        # Final checkpoint with complete statistics
        final_time = time.time()
        final_stats = {
            "avg_time_per_file": (final_time - start_time) / max(1, len(remaining_tasks)),
            "total_elapsed": final_time - start_time,
            "files_completed": completed,
            "success_rate": len(processed_files) / max(1, len(processed_files) + len(failed_files))
        }
        
        checkpoint_manager.save_checkpoint(
            processed_files=list(processed_files),
            failed_files=list(failed_files),
            processing_stats=final_stats
        )
        
        return all_results


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
    
    def _generate_complete_metrics_order(self) -> List[str]:
        """Generate complete metrics list maintaining static order for base metrics."""
        # Static base metrics order (preserved for consistency)
        base_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
        
        # Generate dynamic VCS metrics based on config
        vcs_config = self.config['vcs']
        lct_values = vcs_config['lct']
        chunk_sizes = vcs_config.get('chunk_size', [1])
        if not isinstance(chunk_sizes, list):
            chunk_sizes = [chunk_sizes]  # Convert single value to list
        
        # Create VCS metrics in consistent order: chunk_size ascending, then lct ascending
        vcs_metrics = []
        for chunk_size in sorted(chunk_sizes):
            for lct in sorted(lct_values):
                vcs_metrics.append(f"VCS_C{chunk_size}_LCT{lct}")
        
        return base_metrics + vcs_metrics
    
    def process_single_file(self, json_file: str, output_folder: str) -> List[EvaluationResult]:
        """Process a single JSON file and return results."""
        start_time = time.time()
        file_results = []
        
        try:
            # Load and parse JSON file
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            reference_text = data.get("ground_truth", "")
            if not reference_text:
                if self.logger:
                    self.logger.log_error(f"No ground_truth in file", 
                                        ValueError("Missing ground_truth"), 
                                        {"file_path": json_file})
                return []
            
            # Extract test cases
            test_cases = []
            for category in data.get("categories", []):
                for tc in category.get("test_cases", []):
                    test_cases.append((
                        tc.get("id", ""),
                        tc.get("name", ""),
                        tc.get("description", "")
                    ))
            
            if self.logger:
                self.logger.log_file_start(json_file, len(test_cases))
            
            # Process each test case
            for test_id, test_name, generated_text in test_cases:
                if not generated_text:
                    continue
                
                result = self.evaluate_single_case(
                    reference=reference_text,
                    generated=generated_text,
                    test_case_id=test_id,
                    test_case_name=test_name,
                    file_path=json_file
                )
                file_results.append(result)
            
            # Save individual file results
            if file_results:
                individual_results_dir = self.config['paths'].get('individual_results_dir', 'individual_results')
                individual_output_folder = Path(output_folder) / individual_results_dir
                individual_output_folder.mkdir(parents=True, exist_ok=True)
                self._save_file_results(file_results, Path(json_file).stem, individual_output_folder)
            
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.log_file_complete(json_file, processing_time, len(file_results))
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error processing file: {json_file}", e, {"file_path": json_file})
        
        return file_results
    
    def evaluate_single_case(
        self, 
        reference: str, 
        generated: str, 
        test_case_id: str, 
        test_case_name: str,
        file_path: str = ""
    ) -> EvaluationResult:
        """
        Evaluate a single test case against reference with timing.
        
        Args:
            reference: Ground truth text
            generated: Generated/candidate text
            test_case_id: Unique identifier for test case
            test_case_name: Human-readable test case name
            file_path: Source file path for logging
            
        Returns:
            EvaluationResult containing all computed metrics
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
                chunk_sizes = [chunk_sizes]  # Convert single value to list
            context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
            context_window = vcs_config.get('context_window_control', 4.0)
            
            # Compute VCS metrics for each chunk size and LCT value combination
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
                            lct=lct,  # Pass LCT parameter
                            return_all_metrics=True,
                            return_internals=False
                        )
                        vcs_metrics[f"VCS_C{chunk_size}_LCT{lct}"] = vcs_results.get("VCS", 0.0)
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(f"VCS computation failed for LCT={lct}", e)
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
                **vcs_metrics  # Add all VCS_LCT metrics
            }
            
            # Log test case metrics
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.log_test_case(test_case_id, test_case_name, file_path, 
                                        processing_time, metrics)
            
            return EvaluationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                metrics=metrics
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error evaluating test case: {test_case_id}", e, 
                                    {"test_case_id": test_case_id, "file_path": file_path})
            
            # Return zero metrics on failure
            complete_metrics = self._generate_complete_metrics_order()
            zero_metrics = {metric: 0.0 for metric in complete_metrics}
            return EvaluationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                metrics=zero_metrics
            )
    
    def _save_file_results(self, results: List[EvaluationResult], filename: str, output_dir: Path):
        """Save individual file results to CSV with metrics in rows, test cases in columns."""
        if not results:
            return
            
        # Get decimal precision from config
        decimal_precision = self.config.get('output', {}).get('decimal_precision', 3)
        
        # Get all metric names (including VCS_LCT variants)
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())
        
        # Sort metrics for consistent ordering
        traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
        vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
        ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
        
        # Build data directly in the correct format (metrics as rows, test cases as columns)
        data = {"Metric": ordered_metrics}
        
        # Add each test case as a column (use only test_case_name, keep original formatting)
        for result in results:
            col_name = result.test_case_name  # Use just the test case name as-is
            data[col_name] = [
                round(result.metrics.get(metric, 0.0), decimal_precision)
                for metric in ordered_metrics
            ]
        
        df = pd.DataFrame(data)
        csv_path = output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)


class DataLoader:
    """Handles loading and parsing of evaluation datasets."""
    
    @staticmethod
    def load_json_file(filepath: str) -> Optional[Dict]:
        """Load and parse JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return None
    
    @staticmethod
    def extract_test_cases(data: Dict) -> List[Tuple[str, str, str]]:
        """
        Extract test cases from loaded JSON data.
        
        Returns:
            List of (test_case_id, test_case_name, description) tuples
        """
        test_cases = []
        for category in data.get("categories", []):
            for tc in category.get("test_cases", []):
                test_cases.append((
                    tc.get("id", ""),
                    tc.get("name", ""),
                    tc.get("description", "")
                ))
        return test_cases


class ResultsManager:
    """Handles saving and aggregating evaluation results."""
    
    @staticmethod
    def save_individual_results(
        results: List[EvaluationResult], 
        output_path: str,
        metrics_order: List[str] = None
    ) -> None:
        """Save results for individual file to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to DataFrame
        data = []
        for result in results:
            row = {
                "test_case_id": result.test_case_id,
                "test_case_name": result.test_case_name,
                **result.metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        # Round to specified precision
        numeric_cols = [col for col in df.columns if col not in ["test_case_id", "test_case_name"]]
        df[numeric_cols] = df[numeric_cols].round(DECIMAL_PRECISION)
        
        # Ensure column order - use dynamic metrics order if provided, otherwise fall back to static
        if metrics_order:
            ordered_cols = ["test_case_id", "test_case_name"] + metrics_order
        else:
            # Fallback: traditional metrics first, then VCS metrics alphabetically
            all_metrics = [col for col in df.columns if col not in ["test_case_id", "test_case_name"]]
            traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
            vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
            ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
            ordered_cols = ["test_case_id", "test_case_name"] + ordered_metrics
        
        df = df[ordered_cols]
        
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def aggregate_results(all_results: List[List[EvaluationResult]]) -> AggregatedResults:
        """
        Aggregate results across multiple files.
        
        Args:
            all_results: List of result lists from each file
            
        Returns:
            AggregatedResults with means and standard deviations
        """
        # Flatten all results
        flattened = [result for file_results in all_results for result in file_results]
        
        # Group by test case
        grouped = {}
        for result in flattened:
            key = (result.test_case_id, result.test_case_name)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result.metrics)
        
        # Compute statistics
        aggregated_means = {}
        aggregated_stds = {}
        
        for (test_id, test_name), metrics_list in grouped.items():
            # Convert to DataFrame for easier computation
            df = pd.DataFrame(metrics_list)
            means = df.mean().to_dict()
            stds = df.std().to_dict()
            
            aggregated_means[(test_id, test_name)] = means
            aggregated_stds[(test_id, test_name)] = stds
        
        return AggregatedResults(
            means=aggregated_means,
            stds=aggregated_stds,
            n_samples=len(all_results)
        )
    
    @staticmethod
    def save_aggregated_results(
        aggregated: AggregatedResults, 
        output_path: str,
        config: Dict = None
    ) -> None:
        """Save aggregated results with metrics in rows, test cases in columns."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        decimal_precision = config.get('output', {}).get('decimal_precision', 3) if config else 3
        
        # Get all test cases and metrics
        test_cases = list(aggregated.means.keys())
        test_cases.sort()  # Sort for consistent ordering
        
        # Get all metrics from the first test case
        all_metrics = set()
        for means in aggregated.means.values():
            all_metrics.update(means.keys())
        
        # Sort metrics for consistent ordering
        traditional_metrics = ["BLEU-1", "BLEU-4", "METEOR", "ROUGE-1", "ROUGE-4", "ROUGE-L", "ROUGE-Lsum"]
        vcs_metrics = sorted([m for m in all_metrics if m.startswith("VCS")])
        ordered_metrics = [m for m in traditional_metrics if m in all_metrics] + vcs_metrics
        
        # Build data directly in the correct format (metrics as rows, test cases as columns)
        data = {"Metric": ordered_metrics}
        
        for test_id, test_name in test_cases:
            col_name = test_name  # Use just the test case name, keep original formatting
            means = aggregated.means[(test_id, test_name)]
            stds = aggregated.stds[(test_id, test_name)]
            
            # Format as "mean ± std" for each metric
            data[col_name] = [
                f"{means.get(metric, 0.0):.{decimal_precision}f} ± {stds.get(metric, 0.0):.{decimal_precision}f}"
                for metric in ordered_metrics
            ]
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)


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
        # Create comparison-specific logs directory
        comparison_logs_dir = Path(logs_dir) / 'comparison'
        
        # Extract processing settings
        processing = config['processing']
        max_workers = processing.get('max_workers', 4)
        self.resume_from_checkpoint = processing.get('resume_from_checkpoint', True)
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(comparison_logs_dir, exist_ok=True)
        
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
        self.logger = StructuredLogger(str(comparison_logs_dir), self.experiment_id)
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
        
        self.logger.main_logger.info(f"Starting evaluation pipeline")
        self.logger.main_logger.info(f"Input folder: {self.input_folder}")
        self.logger.main_logger.info(f"Output folder: {self.output_folder}")
        self.logger.main_logger.info(f"Experiment ID: {self.experiment_id}")
        
        start_time = time.time()
        
        try:
            # Discover input files from config
            data_dir = Path(self.input_folder)
            json_files = sorted(glob.glob(str(data_dir / "*.json")))
            
            if not json_files:
                self.logger.main_logger.error(f"No JSON files found in {self.input_folder}")
                return
            
            self.logger.main_logger.info(f"Found {len(json_files)} JSON files to process")
            
            # Prepare file tasks for parallel processing
            file_tasks = [(json_file, self.output_folder) for json_file in json_files]
            
            # Process files in parallel with checkpointing
            all_results = self.parallel_processor.process_files_parallel(
                file_tasks, self.evaluator, self.logger, self.checkpoint_manager, self.config
            )
            
            # Aggregate and save final results
            if all_results:
                self.logger.main_logger.info("Aggregating results...")
                aggregated = self.results_manager.aggregate_results(all_results)
                
                aggregated_results_dir = self.config['paths'].get('aggregated_results_dir', 'aggregated_results')
                aggregated_output_folder = Path(self.output_folder) / aggregated_results_dir
                aggregated_output_folder.mkdir(parents=True, exist_ok=True)
                final_output_path = str(aggregated_output_folder / "aggr_comp.csv")
                self.results_manager.save_aggregated_results(aggregated, final_output_path, self.config)
                
                # Calculate statistics
                total_test_cases = sum(len(file_results) for file_results in all_results)
                total_time = time.time() - start_time
                
                self.logger.main_logger.info(f"Evaluation completed successfully!")
                self.logger.main_logger.info(f"Total test cases processed: {total_test_cases}")
                self.logger.main_logger.info(f"Total processing time: {total_time:.2f} seconds")
                if total_test_cases > 0:
                    self.logger.main_logger.info(f"Average time per test case: {total_time/total_test_cases:.3f} seconds")
                else:
                    self.logger.main_logger.info("Average time per test case: N/A (no test cases processed)")
                
                # Log completion summary
                self.logger.main_logger.info(f"Results saved to: {final_output_path}")
                
                # Clean up checkpoint
                self.checkpoint_manager.clear_checkpoint()
                self.logger.main_logger.info("Checkpoint cleaned up")
            else:
                self.logger.main_logger.warning("No results generated")
        
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise
    
    def resume_evaluation(self) -> None:
        """Resume evaluation from checkpoint if available."""
        if self.resume_from_checkpoint:
            checkpoint_state = self.checkpoint_manager.load_checkpoint()
            if checkpoint_state:
                self.logger.main_logger.info("Checkpoint found - resuming evaluation")
                self.run_evaluation()
                return
        
        self.logger.main_logger.info("No checkpoint found - starting fresh evaluation")
        self.run_evaluation()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VCS Evaluation Framework - Comprehensive text generation model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --config config.yaml
  python evaluate.py --config ../configs/production.yaml
  python evaluate.py --config config.yaml --verbose

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
        config = ConfigLoader.load_config(args.config)
        
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
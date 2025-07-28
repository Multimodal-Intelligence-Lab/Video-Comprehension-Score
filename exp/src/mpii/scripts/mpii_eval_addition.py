"""
Comprehensive Evaluation Framework for Addition Experiments with Ablation Studies

This module provides a standardized evaluation framework for addition experiments,
following research and industry best practices. It includes separate ablation studies and
comparison tables for analyzing the impact of different VCS components.

Metrics included:
- Traditional N-gram based: BLEU-1, BLEU-4
- Semantic similarity: METEOR, ROUGE-1, ROUGE-4, ROUGE-L, ROUGE-Lsum
- VCS-based: GAS, LAS, NAS-D, NAS-L, NAS, VCS
- Ablation metrics: Various combinations of base metrics

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
from tabulate import tabulate

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

# Override _validate_config for addition-specific validation
class AdditionConfigLoader(ConfigLoader):
    @staticmethod
    def _validate_config(config: Dict) -> None:
        """Validate required configuration fields for addition experiments."""
        # Call parent validation first
        ConfigLoader._validate_config(config)
        
        # Addition-specific validation
        required_sections = ['addition']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration with base merging and addition-specific validation."""
        # Use parent class method for base config merging
        config = ConfigLoader.load_config(config_path)
        # Add addition-specific validation
        AdditionConfigLoader._validate_config(config)
        return config


@dataclass
class IterationResult:
    """Container for evaluation metrics from a single iteration."""
    iteration_id: str
    file_name: str
    subfolder: str
    ablation_metrics: Optional[Dict[str, float]] = None
    comparison_metrics: Optional[Dict[str, float]] = None


@dataclass
class AggregatedResults:
    """Container for aggregated evaluation results across multiple iterations."""
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
        
        # Main execution logger
        self.main_logger = logging.getLogger(f"main_{self.experiment_id}")
        self.main_logger.setLevel(logging.INFO)
        self.main_logger.handlers.clear()
        
        main_handler = logging.FileHandler(self.log_dir / f"mpii_addition_{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    def log_file_start(self, file_path: str, total_iterations: int):
        """Log start of file processing."""
        self.main_logger.info(f"Processing file: {file_path} ({total_iterations} iterations)")
    
    def log_file_complete(self, file_path: str, processing_time: float, iterations_processed: int):
        """Log completion of file processing."""
        self.main_logger.info(f"Completed file: {file_path} ({iterations_processed} iterations, {processing_time:.2f}s)")
    
    def log_iteration(self, iteration_id: str, file_path: str, processing_time: float, metrics: Dict[str, float]):
        """Log individual iteration metrics."""
        # Simplified logging - only log errors for iterations
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
                             config: Dict = None) -> List[List[IterationResult]]:
        """Process files in parallel with enhanced checkpointing."""
        
        processed_files = set()
        failed_files = set()
        files_since_last_checkpoint = 0
        start_time = time.time()
        
        # Load checkpoint with configuration validation
        checkpoint_state = checkpoint_manager.load_checkpoint(config)
        if checkpoint_state:
            processed_files = set(checkpoint_state.get("processed_files", []))
            failed_files = set(checkpoint_state.get("failed_files", []))
            logger.main_logger.info(f"Resumed from checkpoint: {len(processed_files)} files processed, {len(failed_files)} failed")
        
        # Filter remaining files (exclude both processed and permanently failed)
        remaining_tasks = [task for task in file_tasks 
                          if task[0] not in processed_files and task[0] not in failed_files]
        
        if not remaining_tasks:
            logger.main_logger.info("All files already processed")
            return []  # Return empty list, aggregation will load from checkpoint files
        
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
        
        return []  # Return empty list, aggregation will load from checkpoint files


# ModelInitializer now imported from utils.core


# TextProcessor now imported from utils.core


# EmbeddingGenerator now imported from utils.core


class AblationAnalyzer:
    """Handles ablation studies and metric combinations."""
    
    @staticmethod
    def harmonic_mean_2(a: float, b: float) -> float:
        """Calculate harmonic mean of two values."""
        return (2.0 * a * b / (a + b)) if (a > 0 and b > 0) and (a + b) != 0 else 0.0

    @staticmethod
    def harmonic_mean_3(a: float, b: float, c: float) -> float:
        """Calculate harmonic mean of three values."""
        if a <= 0 or b <= 0 or c <= 0:
            return 0.0
        return 3.0 / ((1.0/a) + (1.0/b) + (1.0/c))

    @staticmethod
    def compute_sas_cas_scaled(sas: float, cas: float) -> float:
        """Compute scaled combination of SAS and CAS."""
        if cas <= 0:
            return 0.0
        val = sas - (1 - cas)
        return (val / cas) if (val > 0) else 0.0
    
    @staticmethod
    def compute_sas_nas_scaled(sas: float, nas: float) -> float:
        """Compute adaptive scaled combination."""
        if sas < nas:
            numerator = sas - (1 - nas)
            denominator = nas
        else:
            numerator = nas - (1 - sas)
            denominator = sas
        return (numerator / denominator) if (numerator > 0 and denominator != 0) else 0.0

    @staticmethod
    def produce_ablation_metrics(vcs_results: Dict) -> Dict[str, float]:
        """
        Produce ablation metrics from VCS results following VCS_addition_ab.py logic.
        
        Args:
            vcs_results: Dictionary containing VCS computation results
            
        Returns:
            Dictionary with ablation metrics in the specified order
        """
        # Extract base metrics using correct names
        gas = vcs_results.get("GAS", 0.0)  # SAS in old code
        las = vcs_results.get("LAS", 0.0)  # CAS in old code  
        nas_d = vcs_results.get("NAS-D", 0.0)
        nas_l = vcs_results.get("NAS-L", 0.0)
        nas = vcs_results.get("NAS", 0.0)
        gas_las_scaled = vcs_results.get("GAS-LAS-Scaled", 0.0)  # SAS-CAS-Scaled
        vcs_score = vcs_results.get("VCS", 0.0)
        
        # Compute additional metrics following VCS_addition_ab.py logic
        nas_las_scaled = AblationAnalyzer.compute_sas_cas_scaled(nas, las)
        gas_nas_l_scaled = AblationAnalyzer.compute_sas_nas_scaled(gas, nas_l)
        gas_nas_d_scaled = AblationAnalyzer.compute_sas_nas_scaled(gas, nas_d)
        gas_nas_scaled = AblationAnalyzer.compute_sas_nas_scaled(gas, nas)
        
        gas_las_s_plus_nas_d = AblationAnalyzer.compute_sas_nas_scaled(gas_las_scaled, nas_d)
        gas_las_s_plus_nas_l = AblationAnalyzer.compute_sas_nas_scaled(gas_las_scaled, nas_l)
        
        # Build ablation results following the exact order from the user's specification
        ablation_results = {
            "GAS": gas,
            "LAS": las,
            "NAS-D": nas_d,
            "NAS-L": nas_l,
            "NAS": nas,
            "NAS\n+LAS(S)": nas_las_scaled,
            "GAS\n+LAS(S)": gas_las_scaled,
            "GAS\n+NAS-L(S)": gas_nas_l_scaled,
            "GAS\n+NAS-D(S)": gas_nas_d_scaled,
            "GAS\n+NAS(S)": gas_nas_scaled,
            "GAS\n+LAS(S)\n+NAS-D(S)": gas_las_s_plus_nas_d,
            "GAS\n+LAS(S)\n+NAS-L(S)": gas_las_s_plus_nas_l,
            "GAS\n+LAS(S)\n+(NAS-D\n+NAS-L)(S)": vcs_score,
        }
        
        return ablation_results


class MetricsEvaluator:
    """Core evaluation engine for computing all metrics with logging."""
    
    def __init__(self, config: Dict, logger: StructuredLogger = None):
        self.config = config
        self.segmenter = TextProcessor.sat_segmenter
        self.embedding_fn = EmbeddingGenerator.nv_embed_embedding_fn
        self.ablation_analyzer = AblationAnalyzer()
        self.logger = logger
    
    def process_single_file(self, json_file: str, output_folder: str, subfolder: str) -> List[IterationResult]:
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
            
            # Extract iterations
            iterations = data.get("iterations", {})
            num_iterations = self.config['addition']['num_iterations']
            
            if self.logger:
                self.logger.log_file_start(json_file, len(iterations))
            
            # Process each iteration
            for i in range(1, num_iterations + 1):
                iter_key = str(i)
                generated_text = iterations.get(iter_key, "")
                if not generated_text:
                    continue
                
                result = self.evaluate_single_iteration(
                    reference=reference_text,
                    generated=generated_text,
                    iteration_id=iter_key,
                    file_name=Path(json_file).stem,
                    subfolder=subfolder,
                    file_path=json_file
                )
                file_results.append(result)
            
            # Save individual file results
            if file_results:
                individual_results_dir = self.config['paths'].get('individual_results_dir', 'individual_results')
                individual_output_folder = Path(output_folder) / individual_results_dir
                
                # Save ablation results if enabled
                if self.config['addition'].get('ablation', False):
                    ablation_folder = individual_output_folder / 'ablation' / subfolder
                    ablation_folder.mkdir(parents=True, exist_ok=True)
                    self._save_file_results(file_results, Path(json_file).stem, ablation_folder, 'ablation')
                
                # Save comparison results if enabled
                if self.config['addition'].get('comparison', False):
                    comparison_folder = individual_output_folder / 'comparison' / subfolder
                    comparison_folder.mkdir(parents=True, exist_ok=True)
                    self._save_file_results(file_results, Path(json_file).stem, comparison_folder, 'comparison')
            
            processing_time = time.time() - start_time
            if self.logger:
                self.logger.log_file_complete(json_file, processing_time, len(file_results))
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error processing file: {json_file}", e, {"file_path": json_file})
        
        return file_results
    
    def evaluate_single_iteration(
        self, 
        reference: str, 
        generated: str, 
        iteration_id: str, 
        file_name: str,
        subfolder: str,
        file_path: str = ""
    ) -> IterationResult:
        """
        Evaluate a single iteration against reference with timing.
        
        Args:
            reference: Ground truth text
            generated: Generated/candidate text
            iteration_id: Unique identifier for iteration
            file_name: Name of the source file
            subfolder: Subfolder name (beginning, middle, end, random)
            file_path: Source file path for logging
            
        Returns:
            IterationResult containing all computed metrics
        """
        start_time = time.time()
        
        try:
            ablation_metrics = None
            comparison_metrics = None
            
            # Extract VCS configuration
            vcs_config = self.config['vcs']
            lct_values = vcs_config['lct']
            chunk_sizes = vcs_config.get('chunk_size', [1])
            if not isinstance(chunk_sizes, list):
                chunk_sizes = [chunk_sizes]  # Convert single value to list for compatibility
            context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
            context_window = vcs_config.get('context_window_control', 4.0)
            
            # Compute VCS metrics (needed for both ablation and comparison)
            vcs_results = None
            # Try first chunk_size and first successful LCT value
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
                        break  # Use first successful LCT value
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(f"VCS computation failed for chunk_size={chunk_size}, LCT={lct}", e)
                        continue
                if vcs_results:
                    break  # Use first successful chunk_size
            
            if vcs_results is None:
                # Fallback to zero metrics
                vcs_results = {
                    "GAS": 0.0, "LAS": 0.0, "NAS": 0.0, "NAS-D": 0.0, "NAS-L": 0.0,
                    "VCS": 0.0, "GAS-LAS-Scaled": 0.0
                }
            
            # Compute ablation metrics if enabled
            if self.config['addition'].get('ablation', False):
                ablation_metrics = self.ablation_analyzer.produce_ablation_metrics(vcs_results)
            
            # Compute comparison metrics if enabled  
            if self.config['addition'].get('comparison', False):
                # Compute traditional metrics (BLEU, METEOR, ROUGE)
                brm_results = evaluate_BRM(reference, generated)
                
                # Create comparison metrics in the same order as mpii_eval.py
                comparison_metrics = {
                    "BLEU-1": brm_results.get("bleu1", 0.0),
                    "BLEU-4": brm_results.get("bleu4", 0.0),
                    "METEOR": brm_results.get("meteor", 0.0),
                    "ROUGE-1": brm_results.get("rouge", {}).get("rouge1", 0.0),
                    "ROUGE-4": brm_results.get("rouge", {}).get("rouge4", 0.0),
                    "ROUGE-L": brm_results.get("rouge", {}).get("rougeL", 0.0),
                    "ROUGE-Lsum": brm_results.get("rouge", {}).get("rougeLsum", 0.0),
                    "VCS": vcs_results.get("VCS", 0.0),
                }
            
            # Log iteration metrics
            processing_time = time.time() - start_time
            if self.logger:
                all_metrics = {}
                if ablation_metrics:
                    all_metrics.update(ablation_metrics)
                if comparison_metrics:
                    all_metrics.update(comparison_metrics)
                self.logger.log_iteration(iteration_id, file_path, processing_time, all_metrics)
            
            return IterationResult(
                iteration_id=iteration_id,
                file_name=file_name,
                subfolder=subfolder,
                ablation_metrics=ablation_metrics,
                comparison_metrics=comparison_metrics
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error evaluating iteration: {iteration_id}", e, 
                                    {"iteration_id": iteration_id, "file_path": file_path})
            
            # Return empty result on failure
            return IterationResult(
                iteration_id=iteration_id,
                file_name=file_name,
                subfolder=subfolder,
                ablation_metrics=None,
                comparison_metrics=None
            )
    
    def _save_file_results(self, results: List[IterationResult], filename: str, output_dir: Path, result_type: str):
        """Save individual file results to CSV with iterations as rows, metrics as columns."""
        if not results:
            return
            
        # Get decimal precision from config
        decimal_precision = self.config.get('output', {}).get('decimal_precision', 3)
        
        # Build data with iterations as rows, metrics as columns
        data = []
        for result in results:
            row = {"Iteration": result.iteration_id}
            
            # Select metrics based on result type
            if result_type == 'ablation' and result.ablation_metrics:
                metrics = result.ablation_metrics
            elif result_type == 'comparison' and result.comparison_metrics:
                metrics = result.comparison_metrics
            else:
                continue
            
            for metric_name, value in metrics.items():
                row[metric_name] = round(value, decimal_precision)
            data.append(row)
        
        if data:  # Only save if we have data
            df = pd.DataFrame(data)
            csv_path = output_dir / f"{filename}.csv"
            df.to_csv(csv_path, index=False)


class ResultsManager:
    """Handles saving and aggregating evaluation results."""
    
    @staticmethod
    def aggregate_results_by_subfolder(all_results: List[List[IterationResult]], result_type: str) -> Dict[str, AggregatedResults]:
        """
        Aggregate results by subfolder.
        
        Args:
            all_results: List of result lists from each file
            result_type: 'ablation' or 'comparison'
            
        Returns:
            Dictionary with subfolder names as keys and AggregatedResults as values
        """
        # Group results by subfolder
        subfolder_results = {}
        for file_results in all_results:
            for result in file_results:
                subfolder = result.subfolder
                if subfolder not in subfolder_results:
                    subfolder_results[subfolder] = []
                
                # Get metrics based on type
                if result_type == 'ablation' and result.ablation_metrics:
                    subfolder_results[subfolder].append((result.iteration_id, result.ablation_metrics))
                elif result_type == 'comparison' and result.comparison_metrics:
                    subfolder_results[subfolder].append((result.iteration_id, result.comparison_metrics))
        
        # Aggregate within each subfolder
        aggregated_by_subfolder = {}
        for subfolder, results in subfolder_results.items():
            # Group by iteration
            iteration_groups = {}
            for iteration_id, metrics in results:
                if iteration_id not in iteration_groups:
                    iteration_groups[iteration_id] = []
                iteration_groups[iteration_id].append(metrics)
            
            # Compute statistics for each iteration
            aggregated_means = {}
            aggregated_stds = {}
            
            for iteration_id, metrics_list in iteration_groups.items():
                df = pd.DataFrame(metrics_list)
                means = df.mean().to_dict()
                stds = df.std().to_dict()
                
                aggregated_means[iteration_id] = means
                aggregated_stds[iteration_id] = stds
            
            aggregated_by_subfolder[subfolder] = AggregatedResults(
                means=aggregated_means,
                stds=aggregated_stds,
                n_samples=len(results)
            )
        
        return aggregated_by_subfolder
    
    @staticmethod
    def save_aggregated_results(
        aggregated_by_subfolder: Dict[str, AggregatedResults], 
        output_dir: str,
        result_type: str,
        config: Dict = None
    ) -> None:
        """Save aggregated results for each subfolder."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        decimal_precision = config.get('output', {}).get('decimal_precision', 3) if config else 3
        
        for subfolder, aggregated in aggregated_by_subfolder.items():
            # Get all metrics from the first iteration
            if not aggregated.means:
                continue
                
            first_iteration = next(iter(aggregated.means.values()))
            all_metrics = list(first_iteration.keys())
            
            # Build data with iterations as rows, metrics as columns
            data = []
            for iteration_id in sorted(aggregated.means.keys(), key=lambda x: int(x)):
                row = {"Iteration": iteration_id}
                means = aggregated.means[iteration_id]
                stds = aggregated.stds[iteration_id]
                
                for metric in all_metrics:
                    mean_val = means.get(metric, 0.0)
                    std_val = stds.get(metric, 0.0)
                    # Format as "mean ± std"
                    row[metric] = f"{mean_val:.{decimal_precision}f} ± {std_val:.{decimal_precision}f}"
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Save CSV only (no HTML as requested)
            if result_type == 'ablation':
                csv_path = output_path / f"aggr_ab_add_{subfolder}.csv"
            else:  # comparison
                csv_path = output_path / f"aggr_comp_add_{subfolder}.csv"
            
            df.to_csv(csv_path, index=False)


class EvaluationPipeline:
    """Enhanced evaluation pipeline for addition experiments with parallel processing."""
    
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
        # Create addition-specific logs directory
        addition_logs_dir = Path(logs_dir) / 'addition'
        
        # Extract processing settings
        processing = config['processing']
        max_workers = processing.get('max_workers', 4)
        self.resume_from_checkpoint = processing.get('resume_from_checkpoint', True)
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(addition_logs_dir, exist_ok=True)
        
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
        self.logger = StructuredLogger(str(addition_logs_dir), self.experiment_id)
        self.checkpoint_manager = CheckpointManager(self.output_folder, self.experiment_id, config_hash)
        self.parallel_processor = ParallelProcessor(max_workers)
        self.evaluator = MetricsEvaluator(config, self.logger)
        self.results_manager = ResultsManager()
        
        # Initialize models
        self.logger.main_logger.info("Initializing models...")
        ModelInitializer.initialize_sat()
        ModelInitializer.initialize_nvembed(config)
        self.logger.main_logger.info("Models initialized successfully")
    
    def _load_iteration_results_from_checkpoint(self) -> List[List[IterationResult]]:
        """Load IterationResult objects from individual results files for addition script."""
        import json
        from pathlib import Path
        
        all_results = []
        individual_results_dir = self.config['paths'].get('individual_results_dir', 'individual_results')
        base_results_dir = Path(self.output_folder) / individual_results_dir
        
        if not base_results_dir.exists():
            self.logger.main_logger.warning(f"Individual results directory not found: {base_results_dir}")
            return []
        
        try:
            # Process each subfolder
            subfolders = self.config['addition']['subfolders']
            for subfolder in subfolders:
                subfolder_dir = base_results_dir / 'comparison' / subfolder  # Use comparison dir as primary source
                
                if not subfolder_dir.exists():
                    self.logger.main_logger.warning(f"Subfolder results not found: {subfolder_dir}")
                    continue
                
                # Process each CSV file in the subfolder
                for csv_file in sorted(subfolder_dir.glob("*.csv")):
                    file_name = csv_file.stem  # Get filename without extension
                    
                    # Create IterationResult objects for this file
                    file_results = []
                    
                    # Read comparison results
                    import pandas as pd
                    try:
                        df_comp = pd.read_csv(csv_file)
                        
                        # Try to read corresponding ablation results
                        ablation_file = base_results_dir / 'ablation' / subfolder / f"{file_name}.csv"
                        df_abl = None
                        if ablation_file.exists():
                            df_abl = pd.read_csv(ablation_file)
                        
                        # Create IterationResult for each iteration
                        for _, row in df_comp.iterrows():
                            iteration_id = str(int(float(row['Iteration'])))  # Convert float to int to string
                            
                            # Build comparison metrics
                            comparison_metrics = {}
                            for col in df_comp.columns:
                                if col != 'Iteration':
                                    comparison_metrics[col] = float(row[col])
                            
                            # Build ablation metrics if available
                            ablation_metrics = {}
                            if df_abl is not None:
                                abl_row = df_abl[df_abl['Iteration'] == row['Iteration']]
                                if not abl_row.empty:
                                    for col in df_abl.columns:
                                        if col != 'Iteration':
                                            ablation_metrics[col] = float(abl_row.iloc[0][col])
                            
                            result = IterationResult(
                                iteration_id=iteration_id,
                                file_name=file_name,
                                subfolder=subfolder,
                                ablation_metrics=ablation_metrics if ablation_metrics else None,
                                comparison_metrics=comparison_metrics
                            )
                            file_results.append(result)
                        
                        if file_results:
                            all_results.append(file_results)
                            
                    except Exception as e:
                        self.logger.log_error(f"Failed to load results from {csv_file}", e)
                        continue
                        
            self.logger.main_logger.info(f"Loaded results from {len(all_results)} files")
            
        except Exception as e:
            self.logger.log_error("Failed to load results from individual files", e)
            return []
        
        return all_results
    
    def run_evaluation(self) -> None:
        """Execute the complete evaluation pipeline for addition experiments."""
        
        self.logger.main_logger.info(f"Starting addition evaluation pipeline")
        self.logger.main_logger.info(f"Input folder: {self.input_folder}")
        self.logger.main_logger.info(f"Output folder: {self.output_folder}")
        self.logger.main_logger.info(f"Experiment ID: {self.experiment_id}")
        
        start_time = time.time()
        
        try:
            subfolders = self.config['addition']['subfolders']
            
            # Discover input files from all subfolders
            file_tasks = []
            for subfolder in subfolders:
                subfolder_path = Path(self.input_folder) / subfolder
                if not subfolder_path.exists():
                    self.logger.main_logger.warning(f"Subfolder not found: {subfolder_path}")
                    continue
                
                json_files = sorted(glob.glob(str(subfolder_path / "*.json")))
                if not json_files:
                    self.logger.main_logger.warning(f"No JSON files found in {subfolder_path}")
                    continue
                
                # Add tasks for parallel processing
                for json_file in json_files:
                    file_tasks.append((json_file, self.output_folder, subfolder))
            
            if not file_tasks:
                self.logger.main_logger.error("No JSON files found to process")
                return
            
            self.logger.main_logger.info(f"Found {len(file_tasks)} files to process")
            
            # Process files in parallel with checkpointing
            all_results = self.parallel_processor.process_files_parallel(
                file_tasks, self.evaluator, self.logger, self.checkpoint_manager, self.config
            )
            
            # Aggregate and save final results
            self.logger.main_logger.info("Aggregating results...")
            
            # Load all results from checkpoint files
            all_results = self._load_iteration_results_from_checkpoint()
            
            if all_results:
                aggregated_results_dir = self.config['paths'].get('aggregated_results_dir', 'aggregated_results')
                
                # Process ablation results if enabled
                if self.config['addition'].get('ablation', False):
                    ablation_output_folder = Path(self.output_folder) / aggregated_results_dir / 'ablation'
                    ablation_output_folder.mkdir(parents=True, exist_ok=True)
                    
                    aggregated_ablation = self.results_manager.aggregate_results_by_subfolder(all_results, 'ablation')
                    self.results_manager.save_aggregated_results(
                        aggregated_ablation, str(ablation_output_folder), 'ablation', self.config
                    )
                    self.logger.main_logger.info(f"Ablation results saved to: {ablation_output_folder}")
                
                # Process comparison results if enabled
                if self.config['addition'].get('comparison', False):
                    comparison_output_folder = Path(self.output_folder) / aggregated_results_dir / 'comparison'
                    comparison_output_folder.mkdir(parents=True, exist_ok=True)
                    
                    aggregated_comparison = self.results_manager.aggregate_results_by_subfolder(all_results, 'comparison')
                    self.results_manager.save_aggregated_results(
                        aggregated_comparison, str(comparison_output_folder), 'comparison', self.config
                    )
                    self.logger.main_logger.info(f"Comparison results saved to: {comparison_output_folder}")
                
                # Calculate statistics
                total_iterations = sum(len(file_results) for file_results in all_results)
                total_time = time.time() - start_time
                
                self.logger.main_logger.info(f"Addition evaluation completed successfully!")
                self.logger.main_logger.info(f"Total iterations processed: {total_iterations}")
                self.logger.main_logger.info(f"Total processing time: {total_time:.2f} seconds")
                if total_iterations > 0:
                    self.logger.main_logger.info(f"Average time per iteration: {total_time/total_iterations:.3f} seconds")
                else:
                    self.logger.main_logger.info("Average time per iteration: N/A (no iterations processed)")
                
                # Clean up checkpoint
                self.checkpoint_manager.clear_checkpoint()
                self.logger.main_logger.info("Checkpoint cleaned up")
            else:
                self.logger.main_logger.warning("No results found in checkpoint files")
        
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VCS Addition Evaluation Framework - Separate ablation and comparison studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mpii_eval_addition.py --config config.yaml
  python mpii_eval_addition.py --config config.yaml --verbose

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
        config = AdditionConfigLoader.load_config(args.config)
        
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
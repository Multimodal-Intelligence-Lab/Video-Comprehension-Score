"""
MPII Ablation Study Framework

This module provides a comprehensive ablation study framework for VCS metrics,
analyzing individual components and their scaled combinations across test cases.

Features:
- Ablation 1: All VCS metrics (return_all_metrics=True)
- Ablation 2: Custom computed ablation metrics with scaled combinations
- Parallel processing with checkpointing
- Structured logging and result organization
- Individual and aggregated results (mean ± std)

Authors: Research Team
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import vcs
from utils.core import (
    ConfigLoader, CheckpointManager, ModelInitializer, 
    TextProcessor, EmbeddingGenerator, AblationUtils,
    ABLATION_1_METRICS_ORDER, ABLATION_2_METRICS_ORDER,
    DECIMAL_PRECISION, DEFAULT_SEGMENTER_FUNCTION
)


@dataclass
class AblationResult:
    """Container for ablation metrics from a single test case."""
    test_case_id: str
    test_case_name: str
    ablation_type: str
    metrics: Dict[str, float]
    file_id: str  # For grouping results by JSON file


@dataclass
class AggregatedResults:
    """Container for aggregated ablation results."""
    ablation_type: str
    means: Dict[str, float]
    stds: Dict[str, float]
    n_samples: int


class StructuredLogger:
    """Enhanced logging system with JSON format for automated analysis."""
    
    def __init__(self, log_dir: str, experiment_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id or f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
        
        main_handler = logging.FileHandler(self.log_dir / f"mpii_{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    def log_experiment_start(self, ablation_type: str, total_test_cases: int):
        """Log start of ablation experiment."""
        self.main_logger.info(f"Starting {ablation_type} ablation study")
        self.main_logger.info(f"Total test cases to process: {total_test_cases}")
    
    def log_experiment_complete(self, ablation_type: str, processing_time: float, 
                               test_cases_processed: int, avg_metrics: Dict):
        """Log completion of ablation experiment."""
        self.main_logger.info(f"Completed {ablation_type} ablation study")
        self.main_logger.info(f"Processed {test_cases_processed} test cases in {processing_time:.2f}s")
        if test_cases_processed > 0:
            self.main_logger.info(f"Average processing time per test case: {processing_time/test_cases_processed:.2f}s")
        else:
            self.main_logger.info("Average processing time per test case: N/A (no test cases processed)")
        self.main_logger.info(f"Key metric averages: {avg_metrics}")
    
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
        if context:
            self.main_logger.error(f"Context: {context}")
    
    def set_start_time(self):
        """Set the start time for progress tracking."""
        self._start_time = time.time()


class AblationEvaluator:
    """Core ablation evaluation engine."""
    
    def __init__(self, config: Dict, logger: StructuredLogger = None):
        self.config = config
        self.logger = logger
        self.embedding_fn = EmbeddingGenerator.nv_embed_embedding_fn
        
        # Extract VCS parameters
        vcs_config = config['vcs']
        # Support both single values and arrays for chunk_size and lct
        chunk_size_config = vcs_config.get('chunk_size', [1])
        self.chunk_sizes = chunk_size_config if isinstance(chunk_size_config, list) else [chunk_size_config]
        
        lct_config = vcs_config.get('lct', [0])
        self.lct_values = lct_config if isinstance(lct_config, list) else [lct_config]
        
        self.context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
        self.context_window = vcs_config.get('context_window_control', 4.0)
        self.return_all_metrics = vcs_config.get('return_all_metrics', True)
        self.return_internals = vcs_config.get('return_internals', False)
    
    def evaluate_test_case_ablation_1(self, test_case: Dict) -> AblationResult:
        """
        Evaluate a single test case for Ablation 1 (all VCS metrics).
        
        Args:
            test_case: Test case data containing reference and generated texts
            
        Returns:
            AblationResult with all VCS metrics
        """
        test_case_id = test_case.get('test_case_id', 'unknown')
        test_case_name = test_case.get('test_case_name', test_case_id)
        
        try:
            # Extract reference and generated text from test case
            reference = test_case.get('reference', '')
            generated = test_case.get('generated', '')
            
            if not reference or not generated:
                raise ValueError("Missing reference or generated text")
            
            # Get segmenter function
            segmenter_fn = TextProcessor.get_segmenter_function(DEFAULT_SEGMENTER_FUNCTION)
            
            # Compute VCS with all metrics for all chunk_size and lct combinations
            all_vcs_results = {}
            for chunk_size in self.chunk_sizes:
                for lct in self.lct_values:
                    vcs_results = vcs.compute_vcs_score(
                        reference_text=reference,
                        generated_text=generated,
                        segmenter_fn=segmenter_fn,
                        embedding_fn_las=self.embedding_fn,
                        embedding_fn_gas=self.embedding_fn,
                        chunk_size=chunk_size,
                        context_cutoff_value=self.context_cutoff,
                        context_window_control=self.context_window,
                        lct=lct,
                        return_all_metrics=True,  # Always True for ablation 1
                        return_internals=self.return_internals
                    )
                    # Store results with standardized naming
                    metric_prefix = f"VCS_C{chunk_size}_LCT{lct}"
                    for metric_name, value in vcs_results.items():
                        if metric_name == "VCS":
                            all_vcs_results[metric_prefix] = value
                        else:
                            all_vcs_results[f"{metric_prefix}_{metric_name}"] = value
            
            # Filter and organize metrics according to ablation 1 order
            # Include all VCS combinations plus traditional ablation metrics
            filtered_metrics = {}
            
            # Add VCS combinations first (standardized naming)
            for metric_name, value in all_vcs_results.items():
                filtered_metrics[metric_name] = float(value)
            
            # Then add traditional ablation metrics if they exist
            # Note: Traditional ablation metrics are typically from first chunk_size/lct combination
            first_chunk = self.chunk_sizes[0]
            first_lct = self.lct_values[0]
            base_vcs_key = f"VCS_C{first_chunk}_LCT{first_lct}"
            
            for metric_name in ABLATION_1_METRICS_ORDER:
                if metric_name not in filtered_metrics:
                    # Try to find it in the base VCS results with suffix
                    extended_key = f"{base_vcs_key}_{metric_name}"
                    if extended_key in all_vcs_results:
                        filtered_metrics[metric_name] = float(all_vcs_results[extended_key])
                    else:
                        if self.logger:
                            self.logger.main_logger.warning(f"Metric {metric_name} not found in VCS results")
                        filtered_metrics[metric_name] = 0.0
            
            return AblationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                ablation_type="ablation_1",
                metrics=filtered_metrics,
                file_id=test_case.get('file_id', 'unknown')
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error evaluating test case {test_case_id} for ablation 1", e)
            
            # Return zero metrics on failure
            zero_metrics = {metric: 0.0 for metric in ABLATION_1_METRICS_ORDER}
            return AblationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                ablation_type="ablation_1",
                metrics=zero_metrics,
                file_id=test_case.get('file_id', 'unknown')
            )
    
    def evaluate_test_case_ablation_2(self, test_case: Dict) -> AblationResult:
        """
        Evaluate a single test case for Ablation 2 (custom ablation metrics).
        
        Args:
            test_case: Test case data containing reference and generated texts
            
        Returns:
            AblationResult with custom ablation metrics
        """
        test_case_id = test_case.get('test_case_id', 'unknown')
        test_case_name = test_case.get('test_case_name', test_case_id)
        
        try:
            # Extract reference and generated text from test case
            reference = test_case.get('reference', '')
            generated = test_case.get('generated', '')
            
            if not reference or not generated:
                raise ValueError("Missing reference or generated text")
            
            # Get segmenter function
            segmenter_fn = TextProcessor.get_segmenter_function(DEFAULT_SEGMENTER_FUNCTION)
            
            # Compute VCS with all metrics for all chunk_size and lct combinations to get base components
            all_vcs_results = {}
            for chunk_size in self.chunk_sizes:
                for lct in self.lct_values:
                    vcs_results = vcs.compute_vcs_score(
                        reference_text=reference,
                        generated_text=generated,
                        segmenter_fn=segmenter_fn,
                        embedding_fn_las=self.embedding_fn,
                        embedding_fn_gas=self.embedding_fn,
                        chunk_size=chunk_size,
                        context_cutoff_value=self.context_cutoff,
                        context_window_control=self.context_window,
                        lct=lct,
                        return_all_metrics=True,
                        return_internals=self.return_internals
                    )
                    # Store results with standardized naming
                    metric_prefix = f"VCS_C{chunk_size}_LCT{lct}"
                    for metric_name, value in vcs_results.items():
                        if metric_name == "VCS":
                            all_vcs_results[metric_prefix] = value
                        else:
                            all_vcs_results[f"{metric_prefix}_{metric_name}"] = value
            
            # Extract base metrics for ablation 2 computation (use first combination)
            first_chunk = self.chunk_sizes[0]
            first_lct = self.lct_values[0]
            base_prefix = f"VCS_C{first_chunk}_LCT{first_lct}"
            
            gas = all_vcs_results.get(f"{base_prefix}_GAS", 0.0)
            las = all_vcs_results.get(f"{base_prefix}_LAS", 0.0)
            nas_d = all_vcs_results.get(f"{base_prefix}_NAS-D", 0.0)
            nas_l = all_vcs_results.get(f"{base_prefix}_NAS-L", 0.0)
            nas = all_vcs_results.get(f"{base_prefix}_NAS", 0.0)
            
            # Compute ablation 2 metrics using utility function
            ablation_2_metrics = AblationUtils.compute_ablation_2_metrics(
                gas=gas, las=las, nas_d=nas_d, nas_l=nas_l, nas=nas
            )
            
            # Combine traditional ablation 2 metrics with all VCS combinations
            combined_metrics = {}
            combined_metrics.update(ablation_2_metrics)
            combined_metrics.update(all_vcs_results)
            
            return AblationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                ablation_type="ablation_2",
                metrics=combined_metrics,
                file_id=test_case.get('file_id', 'unknown')
            )
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error evaluating test case {test_case_id} for ablation 2", e)
            
            # Return zero metrics on failure
            zero_metrics = {metric: 0.0 for metric in ABLATION_2_METRICS_ORDER}
            return AblationResult(
                test_case_id=test_case_id,
                test_case_name=test_case_name,
                ablation_type="ablation_2",
                metrics=zero_metrics,
                file_id=test_case.get('file_id', 'unknown')
            )


class AblationPipeline:
    """Main ablation evaluation pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Extract paths from config
        paths = config['paths']
        
        # Handle relative paths (script is in scripts/ subdirectory)
        script_dir = Path(__file__).parent.parent
        
        # Data directory
        data_dir_path = Path(paths['data_dir'])
        self.data_dir = str(data_dir_path if data_dir_path.is_absolute() else script_dir / data_dir_path)
        
        # Output folder
        results_dir = Path(paths['results_dir'])
        self.output_folder = str(results_dir if results_dir.is_absolute() else script_dir / results_dir)
        
        # Logs directory 
        logs_dir_path = Path(paths.get('logs_dir', 'logs'))
        ablation_logs_dir = str(logs_dir_path if logs_dir_path.is_absolute() else script_dir / logs_dir_path)
        
        # Extract processing settings
        processing = config['processing']
        self.max_workers = processing.get('max_workers', 4)
        self.resume_from_checkpoint = processing.get('resume_from_checkpoint', True)
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(ablation_logs_dir, exist_ok=True)
        
        # Intelligent experiment ID handling
        experiment_config = config.get('experiment', {})
        config_experiment_id = experiment_config.get('experiment_id')
        
        if not config_experiment_id or config_experiment_id.strip() == "":
            # Case 1: No experiment ID in config → start fresh with new ID
            self.experiment_id = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
                self.experiment_id = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"No checkpoint found for experiment ID: {config_experiment_id}")
                print(f"Starting fresh evaluation with new experiment ID: {self.experiment_id}")
        
        # Generate configuration hash
        import hashlib
        config_copy = config.copy()
        config_copy.pop('experiment', None)
        config_str = json.dumps(config_copy, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Initialize components
        self.logger = StructuredLogger(ablation_logs_dir, self.experiment_id)
        self.checkpoint_manager = CheckpointManager(self.output_folder, self.experiment_id, config_hash)
        self.evaluator = AblationEvaluator(config, self.logger)
        
        # Initialize models
        self.logger.main_logger.info("Initializing models...")
        ModelInitializer.initialize_nvembed(config)
        ModelInitializer.initialize_sat(config)
        self.logger.main_logger.info("Models initialized successfully")
    
    def run_ablation_study(self) -> None:
        """Execute the complete ablation study pipeline."""
        
        self.logger.main_logger.info(f"Starting ablation study pipeline")
        self.logger.main_logger.info(f"Data directory: {self.data_dir}")
        self.logger.main_logger.info(f"Output folder: {self.output_folder}")
        self.logger.main_logger.info(f"Experiment ID: {self.experiment_id}")
        
        start_time = time.time()
        
        try:
            # Load test cases
            self.logger.main_logger.info(f"Loading test cases from: {self.data_dir}")
            test_cases = AblationUtils.extract_test_cases(self.data_dir)
            
            # Check which ablation types to run
            ablation_config = self.config.get('ablation', {})
            run_ablation_1 = ablation_config.get('run_ablation_1', True)
            run_ablation_2 = ablation_config.get('run_ablation_2', True)
            
            # Run ablation studies
            if run_ablation_1:
                self.logger.main_logger.info("Running Ablation 1 (all VCS metrics)")
                ablation_1_results = self._run_ablation_type("ablation_1", test_cases)
                self._save_results(ablation_1_results, "ablation_1")
            
            if run_ablation_2:
                self.logger.main_logger.info("Running Ablation 2 (custom ablation metrics)")
                ablation_2_results = self._run_ablation_type("ablation_2", test_cases)
                self._save_results(ablation_2_results, "ablation_2")
            
            total_time = time.time() - start_time
            self.logger.main_logger.info(f"Ablation study completed in {total_time:.2f} seconds")
            
            # Clean up checkpoint files after successful completion
            if self.resume_from_checkpoint:
                self.checkpoint_manager.clear_checkpoint()
            
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise
    
    def _run_ablation_type(self, ablation_type: str, test_cases: List[Dict]) -> List[AblationResult]:
        """Run a specific ablation type on all test cases."""
        
        self.logger.log_experiment_start(ablation_type, len(test_cases))
        experiment_start_time = time.time()
        
        # Check for checkpoint resume
        processed_test_case_ids = []
        if self.resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.config)
            if checkpoint_data:
                # Check if we have processed test cases for this ablation type
                all_tmp_results = self.checkpoint_manager.load_all_results()
                ablation_results = all_tmp_results.get(ablation_type, [])
                
                if ablation_results:
                    # We have results for this ablation type, check if all test cases are processed
                    # Create unique identifiers using file_id + test_case_id to handle duplicate test case IDs across files
                    processed_unique_ids = [f"{r['file_id']}:{r['test_case_id']}" for r in ablation_results]
                    self.logger.main_logger.info(f"Found {len(processed_unique_ids)} {ablation_type} test cases from checkpoint")
                    
                    # Filter out already processed test cases using unique file_id:test_case_id combination
                    original_count = len(test_cases)
                    current_unique_ids = [f"{case.get('file_id', 'unknown')}:{case.get('test_case_id', 'unknown')}" for case in test_cases]
                    self.logger.main_logger.info(f"Current unique IDs (first 5): {current_unique_ids[:5]}")
                    self.logger.main_logger.info(f"Processed unique IDs (first 5): {processed_unique_ids[:5]}")
                    remaining_test_cases = [case for case in test_cases if f"{case.get('file_id', 'unknown')}:{case.get('test_case_id', 'unknown')}" not in processed_unique_ids]
                    
                    # Check if ALL test cases are truly processed (compare with original total)
                    if len(processed_unique_ids) >= original_count:
                        self.logger.main_logger.info(f"All {ablation_type} test cases already processed, loading results from checkpoint")
                        # Generate results from checkpoint data
                        converted_results = []
                        for result_data in ablation_results:
                            ablation_result = AblationResult(
                                test_case_id=result_data.get('test_case_id', ''),
                                test_case_name=result_data.get('test_case_name', ''),
                                ablation_type=result_data.get('ablation_type', ''),
                                metrics=result_data.get('metrics', {}),
                                file_id=result_data.get('file_id', '')
                            )
                            converted_results.append(ablation_result)
                        
                        experiment_time = time.time() - experiment_start_time
                        
                        # Calculate average metrics for logging
                        avg_metrics = {}
                        if converted_results:
                            metric_names = list(converted_results[0].metrics.keys())
                            for metric_name in metric_names:
                                metric_values = [r.metrics[metric_name] for r in converted_results]
                                avg_metrics[metric_name] = np.mean(metric_values)
                        
                        self.logger.log_experiment_complete(
                            ablation_type, experiment_time, len(converted_results), avg_metrics
                        )
                        
                        return converted_results
                    else:
                        # Update test_cases to use the filtered list
                        test_cases = remaining_test_cases
                        self.logger.main_logger.info(f"Resuming {ablation_type}: {len(processed_unique_ids)} completed, {len(test_cases)} remaining")
                elif checkpoint_data.get('current_ablation_type') == ablation_type:
                    # Legacy checkpoint format - fallback
                    processed_test_case_ids = checkpoint_data.get('processed_test_cases', [])
                    self.logger.main_logger.info(f"Resuming {ablation_type} from legacy checkpoint: {len(processed_test_case_ids)} test cases processed")
                    test_cases = [case for case in test_cases if case.get('test_case_id', 'unknown') not in processed_test_case_ids]
        
        # Process test cases with parallel execution
        results = []
        self.logger.set_start_time()
        test_cases_since_last_checkpoint = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            if ablation_type == "ablation_1":
                future_to_case = {
                    executor.submit(self.evaluator.evaluate_test_case_ablation_1, case): case
                    for case in test_cases
                }
            else:  # ablation_2
                future_to_case = {
                    executor.submit(self.evaluator.evaluate_test_case_ablation_2, case): case
                    for case in test_cases
                }
            
            # Process completed tasks
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    test_case_id = result.test_case_id
                    processed_test_case_ids.append(test_case_id)
                    test_cases_since_last_checkpoint += 1
                    
                    # Save individual result to JSON tmp directory grouped by file_id
                    test_case_data = {
                        'test_case_id': result.test_case_id,
                        'test_case_name': result.test_case_name,
                        'metrics': result.metrics
                    }
                    self.checkpoint_manager.save_test_case_results(result.file_id, ablation_type, test_case_data)
                    
                    # Save checkpoint periodically
                    if self.resume_from_checkpoint and self.checkpoint_manager.should_save_checkpoint(test_cases_since_last_checkpoint):
                        self.checkpoint_manager.save_checkpoint(
                            processed_test_cases=processed_test_case_ids,
                            failed_test_cases=[],  # Track failures if needed
                            current_ablation_type=ablation_type,
                            processing_stats={
                                'avg_time_per_case': (time.time() - self.logger._start_time) / len(processed_test_case_ids) if len(processed_test_case_ids) > 0 else 1.0
                            }
                        )
                        test_cases_since_last_checkpoint = 0  # Reset counter after saving checkpoint
                    
                    # Log progress
                    if len(processed_test_case_ids) % 10 == 0 or len(processed_test_case_ids) == len(test_cases):
                        self.logger.log_progress(len(processed_test_case_ids), len(test_cases), f"Processing {ablation_type}")
                        
                except Exception as e:
                    case_id = case.get('test_case_id', 'unknown')
                    self.logger.log_error(f"Failed to process test case {case_id}", e)
        
        experiment_time = time.time() - experiment_start_time
        
        # Calculate average metrics for logging
        avg_metrics = {}
        if results:
            metric_names = list(results[0].metrics.keys())
            for metric_name in metric_names:
                metric_values = [r.metrics[metric_name] for r in results]
                avg_metrics[metric_name] = np.mean(metric_values)
        
        self.logger.log_experiment_complete(
            ablation_type, experiment_time, len(results), avg_metrics
        )
        
        return results
    
    def _save_results(self, results: List[AblationResult], ablation_type: str) -> None:
        """Save results for a specific ablation type, loading from checkpoint tmp directory."""
        
        ablation_dir = Path(self.output_folder) / ablation_type
        individual_results_dir = ablation_dir / "individual_results"
        aggregated_results_dir = ablation_dir / "aggregated_results"
        
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        aggregated_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all results from checkpoint manager's tmp directory
        all_tmp_results = self.checkpoint_manager.load_all_results()
        ablation_results = all_tmp_results.get(ablation_type, [])
        
        # Convert from JSON format back to AblationResult objects
        converted_results = []
        for result_data in ablation_results:
            ablation_result = AblationResult(
                test_case_id=result_data.get('test_case_id', ''),
                test_case_name=result_data.get('test_case_name', ''),
                ablation_type=result_data.get('ablation_type', ''),
                metrics=result_data.get('metrics', {}),
                file_id=result_data.get('file_id', '')
            )
            converted_results.append(ablation_result)
        
        # Use converted results if available, otherwise fall back to passed results
        final_results = converted_results if converted_results else results
        
        # Save individual results (one CSV per test case)
        if self.config['output'].get('save_individual_results', True):
            self._save_individual_results(final_results, str(individual_results_dir), ablation_type)
        
        # Save aggregated results (mean ± std)
        if self.config['output'].get('save_aggregated_results', True):
            self._save_aggregated_results(final_results, str(aggregated_results_dir), ablation_type)
    
    def _save_individual_results(self, results: List[AblationResult], 
                                output_dir: str, ablation_type: str) -> None:
        """Save individual results as CSV files (one per JSON file with all test cases as columns)."""
        
        decimal_precision = self.config['output'].get('decimal_precision', DECIMAL_PRECISION)
        
        # Group results by file_id (which JSON file they came from)
        file_groups = {}
        for result in results:
            # Get the file_id stored in the test case data 
            file_id = getattr(result, 'file_id', 'unknown')
            if file_id not in file_groups:
                file_groups[file_id] = []
            file_groups[file_id].append(result)
        
        # Create one CSV per JSON file
        for file_id, file_results in file_groups.items():
            # Determine metrics order based on ablation type
            if ablation_type == "ablation_1":
                ordered_metrics = ABLATION_1_METRICS_ORDER
            else:  # ablation_2
                ordered_metrics = ABLATION_2_METRICS_ORDER
            
            # Only use metrics that actually exist in the results
            available_metrics = set()
            for result in file_results:
                available_metrics.update(result.metrics.keys())
            final_metrics = [m for m in ordered_metrics if m in available_metrics]
            
            # Sort results by test_case_id to maintain original JSON file order
            file_results_sorted = sorted(file_results, key=lambda x: x.test_case_id)
            
            # Create CSV with metrics as rows, test cases as columns
            csv_data = []
            for metric_name in final_metrics:
                row = {'metric': metric_name}
                for result in file_results_sorted:
                    metric_value = result.metrics.get(metric_name, 0.0)
                    row[result.test_case_name] = round(metric_value, decimal_precision)
                csv_data.append(row)
            
            # Save CSV file for this JSON file
            df = pd.DataFrame(csv_data)
            output_path = Path(output_dir) / f"{file_id}.csv"
            df.to_csv(output_path, index=False)
        
        self.logger.main_logger.info(f"Saved {len(file_groups)} individual {ablation_type} CSV files to: {output_dir}")
    
    def _save_aggregated_results(self, results: List[AblationResult], 
                                output_dir: str, ablation_type: str) -> None:
        """Save aggregated results with mean ± std for each metric-testcase combination across JSON files."""
        
        if not results:
            return
        
        decimal_precision = self.config['output'].get('decimal_precision', DECIMAL_PRECISION)
        
        # Group results by test case name and collect values for each metric across JSON files
        test_case_metrics = {}
        
        for result in results:
            test_case_name = result.test_case_name
            if test_case_name not in test_case_metrics:
                test_case_metrics[test_case_name] = {}
            
            for metric_name, metric_value in result.metrics.items():
                if metric_name not in test_case_metrics[test_case_name]:
                    test_case_metrics[test_case_name][metric_name] = []
                test_case_metrics[test_case_name][metric_name].append(metric_value)
        
        # Determine metrics order based on ablation type
        if ablation_type == "ablation_1":
            ordered_metrics = ABLATION_1_METRICS_ORDER
        else:  # ablation_2
            ordered_metrics = ABLATION_2_METRICS_ORDER
        
        # Only use metrics that actually exist in the results
        available_metrics = set()
        for test_case_data in test_case_metrics.values():
            available_metrics.update(test_case_data.keys())
        final_metrics = [m for m in ordered_metrics if m in available_metrics]
        
        # Sort test cases by test_case_id to maintain original JSON file order
        # Convert test case names back to IDs for proper sorting, then back to names
        test_case_id_to_name = {}
        for result in results:
            test_case_id_to_name[result.test_case_id] = result.test_case_name
        
        # Sort by test_case_id and get corresponding names in order
        sorted_test_case_ids = sorted(test_case_id_to_name.keys(), 
                                    key=lambda x: [int(part) if part.isdigit() else part 
                                                 for part in x.replace('.', ' ').split()])
        all_test_cases = [test_case_id_to_name[tid] for tid in sorted_test_case_ids 
                         if test_case_id_to_name[tid] in test_case_metrics]
        
        # Create aggregated CSV with metrics as rows, test cases as columns
        csv_data = []
        for metric_name in final_metrics:
            row = {'Metric': metric_name}
            
            for test_case_name in all_test_cases:
                if (test_case_name in test_case_metrics and 
                    metric_name in test_case_metrics[test_case_name]):
                    values = test_case_metrics[test_case_name][metric_name]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    row[test_case_name] = f"{round(mean_val, decimal_precision)} ± {round(std_val, decimal_precision)}"
                else:
                    # Missing data
                    row[test_case_name] = "0.000 ± 0.000"
            
            csv_data.append(row)
        
        # Save aggregated CSV
        df = pd.DataFrame(csv_data)
        output_path = Path(output_dir) / "aggregated_results.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.main_logger.info(f"Saved {ablation_type} aggregated results to: {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MPII Ablation Study Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml
  python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --experiment_id my_exp
  python scripts/mpii_eval_ablation.py --config config/mpii_eval_ablation.yaml --verbose

For detailed configuration options, see config/mpii_eval_ablation.yaml file.
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to YAML configuration file'
    )
    
    parser.add_argument(
        '--experiment_id', 
        type=str,
        help='Override experiment ID from config'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output (overrides config setting)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        config = ConfigLoader.load_config(str(config_path))
        
        # Override experiment ID if provided
        if args.experiment_id:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['experiment_id'] = args.experiment_id
        
        # Override verbosity if specified
        if args.verbose:
            config.setdefault('logging', {})['verbose'] = True
        
        # Initialize and run pipeline
        pipeline = AblationPipeline(config)
        pipeline.run_ablation_study()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
"""
VLM Evaluation Framework for CLIP-CC Dataset

This module provides a comprehensive evaluation framework for evaluating Vision-Language Models (VLMs)
against the CLIP-CC dataset using VCS (Video Comprehension Score) metrics.

Features:
- Evaluation of multiple VLMs against CLIP-CC ground truth summaries
- VCS computation for multiple chunk sizes and LCT values
- Parallel processing for efficiency
- Checkpointing for resuming interrupted evaluations
- Structured CSV output for individual and aggregated results

Authors: Research Team
"""

import torch
import os
import json
import pandas as pd
import numpy as np
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import vcs
from utils.core import (
    ConfigLoader, DataLoader, VCSEvaluator, ResultsProcessor, 
    ModelInitializer, VLMEvaluationResult, TextProcessor, EmbeddingGenerator, CheckpointManager
)

# Import global model instances from utils
from utils.core import sat_adapted, model_nv, device_embed


class StructuredLogger:
    """Enhanced logging system for VLM evaluation."""
    
    def __init__(self, log_dir: str, experiment_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup loggers
        self.setup_loggers()
        
    def setup_loggers(self):
        """Setup structured logging like mpii - avoid library warnings in log files."""
        log_file = self.log_dir / f"clipcc_{self.experiment_id}.log"
        
        # Clear root logger handlers to prevent library warnings from being logged
        logging.getLogger().handlers.clear()
        
        # Create specific named logger like mpii
        self.logger = logging.getLogger(f"main_{self.experiment_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Create file handler for detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Create console handler for only critical messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)  # Only show errors on terminal
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def log_evaluation_start(self, config: Dict, models_info: Dict):
        """Log the start of evaluation with configuration details."""
        self.logger.info("="*80)
        self.logger.info("VLM EVALUATION ON CLIP-CC DATASET")
        self.logger.info("="*80)
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        self.logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        self.logger.info(f"Models to evaluate: {list(models_info.keys())}")
        self.logger.info(f"Chunk sizes: {config['vcs']['chunk_size']}")
        self.logger.info(f"LCT values: {config['vcs']['lct']}")
        self.logger.info(f"Parallel workers: {config['processing']['max_workers']}")
        self.logger.info("="*80)
    
    def log_model_progress(self, model_name: str, current: int, total: int, duration: float = None, 
                          processing_stats: Dict = None):
        """Log detailed progress for model evaluation like mpii."""
        # Basic progress with percentage
        percentage = (current / total * 100) if total > 0 else 0
        progress_msg = f"Model {model_name}: {current}/{total} ({percentage:.1f}%) samples"
        
        if duration:
            progress_msg += f" (took {duration:.2f}s)"
            
        self.logger.info(progress_msg)
        
        # Add ETA calculation and performance metrics if available
        if processing_stats and current > 0:
            avg_time = processing_stats.get("avg_time_per_file", 0)
            if avg_time > 0:
                remaining = total - current
                eta_seconds = remaining * avg_time
                eta_minutes = eta_seconds / 60
                self.logger.info(f"  Performance: {avg_time:.2f}s per sample, ETA: {eta_minutes:.1f} minutes")
                
            total_elapsed = processing_stats.get("total_elapsed", 0)
            if total_elapsed > 0:
                self.logger.info(f"  Total elapsed: {total_elapsed/60:.1f} minutes")
    
    def log_checkpoint_event(self, event_type: str, interval: int = None, files_processed: int = None):
        """Log checkpoint events like mpii."""
        msg = f"Checkpoint {event_type}"
        if interval:
            msg += f" (interval: {interval})"
        if files_processed:
            msg += f" after {files_processed} files"
        self.logger.info(msg)
    
    def log_evaluation_complete(self, total_time: float, total_models: int, total_samples: int):
        """Log completion of evaluation."""
        self.logger.info("="*80)
        self.logger.info("EVALUATION COMPLETED")
        self.logger.info(f"Total time: {total_time:.2f} seconds")
        self.logger.info(f"Models evaluated: {total_models}")
        self.logger.info(f"Total samples: {total_samples}")
        if total_models > 0:
            self.logger.info(f"Average time per model: {total_time/total_models:.2f}s")
        else:
            self.logger.info("Average time per model: N/A (no models processed)")
        self.logger.info("="*80)


class ParallelVLMEvaluator:
    """Parallel evaluation system for VLMs with checkpointing."""
    
    def __init__(self, config: Dict, logger: StructuredLogger):
        self.config = config
        self.logger = logger
        # Use experiment_id from config (already determined by main function logic)
        self.experiment_id = config['experiment']['experiment_id']
        
        # Setup results directories
        self.setup_directories()
        
        # Initialize checkpointing - use mpii approach with enhanced features
        results_dir = Path(self.config['paths']['results_dir'])
        self.checkpoint_manager = CheckpointManager(str(results_dir), self.experiment_id, self.config, self.logger.logger)
        self.processed_models = set()
        
        # Initialize VCS evaluator with checkpoint manager for incremental saving
        self.vcs_evaluator = VCSEvaluator(config, self.checkpoint_manager, self.logger.logger)
        
    def setup_directories(self):
        """Create necessary directories for results."""
        results_dir = Path(self.config['paths']['results_dir'])
        self.individual_results_dir = results_dir / self.config['paths']['individual_results_dir']
        self.aggregated_results_dir = results_dir / self.config['paths']['aggregated_results_dir']
        
        self.individual_results_dir.mkdir(parents=True, exist_ok=True)
        self.aggregated_results_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_all_models(self, models_data: Dict[str, Dict], ground_truth: Dict[str, Dict]) -> Dict[str, List[VLMEvaluationResult]]:
        """
        Evaluate all models in parallel against ground truth.
        
        Args:
            models_data: Dictionary mapping model names to their predictions
            ground_truth: Ground truth data from CLIP-CC
            
        Returns:
            Dictionary mapping model names to their evaluation results
        """
        all_results = {}
        
        # Load checkpoint if exists - get detailed summary-level tracking like mpii
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        total_summaries_expected = sum(len(data) for data in models_data.values())
        
        if checkpoint_data:
            self.processed_models = set(checkpoint_data.get("processed_models", []))
            # Load previously saved results
            all_results = self.checkpoint_manager.load_all_results()
            
            # Get detailed checkpoint info like mpii
            processed_files = self.checkpoint_manager.processed_files if hasattr(self.checkpoint_manager, 'processed_files') else set()
            failed_files = self.checkpoint_manager.failed_files if hasattr(self.checkpoint_manager, 'failed_files') else set()
            processing_stats = self.checkpoint_manager.processing_stats if hasattr(self.checkpoint_manager, 'processing_stats') else {}
            
            # Count processed summaries from loaded results
            total_processed_summaries = sum(len(results) for results in all_results.values())
            
            # Show detailed resumption info like mpii
            if total_processed_summaries > 0 or len(processed_files) > 0:
                self.logger.logger.info(f"Resumed from checkpoint: {total_processed_summaries} summaries processed ({len(self.processed_models)} models completed)")
                if len(failed_files) > 0:
                    self.logger.logger.info(f"  - {len(failed_files)} summaries failed in previous runs")
                if processing_stats.get('total_elapsed'):
                    elapsed_minutes = processing_stats['total_elapsed'] / 60
                    self.logger.logger.info(f"  - Previous processing time: {elapsed_minutes:.1f} minutes")
            else:
                self.logger.logger.info("No previous progress found, starting fresh evaluation")
        else:
            self.logger.logger.info("No checkpoint file found, starting fresh evaluation")
        
        max_workers = self.config['processing']['max_workers']
        
        # Filter models to process (skip already processed ones)
        models_to_process = {name: data for name, data in models_data.items() 
                           if name not in self.processed_models}
        
        if not models_to_process:
            self.logger.logger.info("All models already processed")
            return all_results
        
        # Calculate actual remaining summaries based on loaded results (more reliable than checkpoint files list)
        total_processed_summaries = sum(len(results) for results in all_results.values())
        total_remaining = total_summaries_expected - total_processed_summaries
        
        # Debug: Log the calculation
        self.logger.logger.info(f"Debug: Total expected: {total_summaries_expected}, Already loaded results: {total_processed_summaries}, Remaining: {total_remaining}")
        
        self.logger.logger.info(f"Processing {total_remaining} remaining summaries across {len(models_to_process)} models with {max_workers} workers")
        if total_processed_summaries > 0:
            if total_summaries_expected > 0:
                self.logger.logger.info(f"Progress: {total_processed_summaries}/{total_summaries_expected} summaries already completed ({total_processed_summaries/total_summaries_expected*100:.1f}%)")
            else:
                self.logger.logger.info(f"Progress: {total_processed_summaries}/0 summaries already completed (N/A%)")
        
        # Process models in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit evaluation tasks
            future_to_model = {}
            for model_name, predictions in models_to_process.items():
                future = executor.submit(
                    self._evaluate_single_model,
                    model_name,
                    predictions,
                    ground_truth
                )
                future_to_model[future] = model_name
            
            # Collect results as they complete
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    start_time = time.time()
                    results = future.result()
                    duration = time.time() - start_time
                    
                    all_results[model_name] = results
                    self.processed_models.add(model_name)
                    
                    # Enhanced logging with running totals like mpii
                    current_total_summaries = sum(len(r) for r in all_results.values())
                    
                    processing_stats = self.checkpoint_manager.processing_stats.copy()
                    processing_stats.update({
                        "total_models": len(models_data),
                        "completed_models": len(self.processed_models),
                        "total_summaries_expected": total_summaries_expected,
                        "current_total_summaries": current_total_summaries,
                        "last_processed": model_name,
                        "processing_time": duration
                    })
                    
                    # Log progress like mpii: current_processed/total (percentage)
                    completion_pct = (current_total_summaries / total_summaries_expected * 100) if total_summaries_expected > 0 else 0
                    self.logger.logger.info(f"Completed {model_name}: {len(results)} summaries in {duration:.1f}s")
                    self.logger.logger.info(f"Overall progress: {current_total_summaries}/{total_summaries_expected} summaries ({completion_pct:.1f}%)")
                    
                    # ETA calculation
                    if current_total_summaries > total_processed_summaries and duration > 0:  # Only if we've made progress
                        remaining = total_summaries_expected - current_total_summaries
                        avg_time_per_summary = processing_stats.get("avg_time_per_file", duration / len(results))
                        eta_minutes = (remaining * avg_time_per_summary) / 60
                        if eta_minutes > 0:
                            self.logger.logger.info(f"ETA: {eta_minutes:.1f} minutes remaining")
                    
                    # Log checkpoint event
                    files_processed = self.checkpoint_manager.processing_stats.get("files_completed", 0)
                    self.logger.log_checkpoint_event(
                        "saved", 
                        self.checkpoint_manager._adaptive_interval,
                        files_processed
                    )
                    
                    # Individual results are saved automatically by interval-based checkpoint manager
                    
                    # Save to checkpoint system for recovery
                    self.checkpoint_manager.save_model_results(model_name, results)
                    
                    # Update main checkpoint - ensure processed files are included from VCS evaluator
                    # The individual processed files are tracked by the VCS evaluator's checkpoint manager
                    # So we need to make sure they're accessible here
                    self.checkpoint_manager.save_checkpoint(list(self.processed_models), processing_stats)
                    
                except Exception as e:
                    self.logger.logger.error(f"Error evaluating model {model_name}: {e}")
                    all_results[model_name] = []
        
        return all_results
    
    def _evaluate_single_model(
        self, 
        model_name: str, 
        predictions: Dict[str, Dict], 
        ground_truth: Dict[str, Dict]
    ) -> List[VLMEvaluationResult]:
        """Evaluate a single model against ground truth."""
        try:
            results = self.vcs_evaluator.evaluate_model(
                model_name, 
                predictions, 
                ground_truth
            )
            return results
        except Exception as e:
            self.logger.logger.error(f"Failed to evaluate model {model_name}: {e}")
            return []
    


def initialize_models(config: Dict) -> None:
    """Initialize all required models for VCS computation."""
    print("Initializing models...")
    
    # Initialize SAT model for text segmentation
    ModelInitializer.initialize_sat(config['models']['sat_model'])
    print("✓ SAT model initialized")
    
    # Initialize NV-Embed model for embeddings
    ModelInitializer.initialize_nvembed(config)
    print("✓ NV-Embed model initialized")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate VLMs on CLIP-CC dataset using VCS metrics"
    )
    parser.add_argument(
        '--config', 
        type=str,
        default='../config/clipcc_eval_vlms.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--experiment_id',
        type=str,
        help='Override experiment ID from config'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_path = Path(args.config)
        config = ConfigLoader.load_config(str(config_path))
        
        # Override experiment ID if provided
        if args.experiment_id:
            if 'experiment' not in config:
                config['experiment'] = {}
            config['experiment']['experiment_id'] = args.experiment_id
        
        # Intelligent experiment ID handling
        config_experiment_id = config.get('experiment', {}).get('experiment_id')
        script_dir = Path(__file__).parent.parent
        logs_dir = script_dir / "logs"
        results_dir = Path(config['paths']['results_dir'])
        
        if not config_experiment_id or config_experiment_id.strip() == "":
            # Case 1: No experiment ID in config → start fresh with new ID
            experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            config['experiment'] = config.get('experiment', {})
            config['experiment']['experiment_id'] = experiment_id
            logger = StructuredLogger(str(logs_dir), experiment_id)
            print(f"Starting fresh evaluation with new experiment ID: {experiment_id}")
        else:
            # Case 2: Experiment ID provided → check for checkpoint files
            experiment_id = config_experiment_id.strip()
            checkpoint_file = results_dir / f"checkpoint_{experiment_id}.json.gz"
            
            if checkpoint_file.exists():
                # Resume from existing checkpoint + use existing log file
                logger = StructuredLogger(str(logs_dir), experiment_id)
                print(f"Found checkpoint for experiment ID: {experiment_id}, will attempt to resume")
            else:
                # No checkpoint found → start fresh with new ID + new log file
                new_experiment_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                config['experiment']['experiment_id'] = new_experiment_id
                experiment_id = new_experiment_id
                logger = StructuredLogger(str(logs_dir), experiment_id)
                print(f"No checkpoint found for experiment ID: {config_experiment_id}")
                print(f"Starting fresh evaluation with new experiment ID: {experiment_id}")
        
        # Load data
        print("Loading CLIP-CC ground truth data...")
        ground_truth = DataLoader.load_clipcc_data(config['paths']['clipcc_data_dir'])
        
        print("Loading model predictions...")
        models_data = DataLoader.load_model_predictions(config['paths']['models_data_dir'])
        
        # Log evaluation start
        logger.log_evaluation_start(config, models_data)
        
        # Initialize models
        initialize_models(config)
        
        # Create evaluator
        evaluator = ParallelVLMEvaluator(config, logger)
        
        # Run evaluation
        print("Starting parallel evaluation...")
        start_time = time.time()
        
        all_results = evaluator.evaluate_all_models(models_data, ground_truth)
        
        total_time = time.time() - start_time
        
        # Compute and save aggregated results
        logger.logger.info("Computing aggregated results...")
        aggregated_output = evaluator.aggregated_results_dir / "aggregated_results.csv"
        ResultsProcessor.compute_aggregated_results(
            all_results,
            str(aggregated_output),
            config['output']['decimal_precision']
        )
        logger.logger.info(f"Saved aggregated results to {aggregated_output}")
        
        # Calculate total samples
        total_samples = sum(len(results) for results in all_results.values())
        
        # Log completion
        logger.log_evaluation_complete(total_time, len(models_data), total_samples)
        
        # Clear checkpoints since evaluation completed successfully
        evaluator.checkpoint_manager.clear_checkpoint()
        
        # Final summary logged to file
        logger.logger.info(f"Evaluation completed successfully!")
        logger.logger.info(f"Total time: {total_time:.2f} seconds") 
        logger.logger.info(f"Results saved to: {evaluator.individual_results_dir}")
        logger.logger.info(f"Aggregated results: {aggregated_output}")
        logger.logger.info(f"Checkpoints cleared.")
        
        # Brief terminal output
        print(f"\\nEvaluation completed! Check log file for details.")
        print(f"Results: {evaluator.individual_results_dir}")
        print(f"Aggregated: {aggregated_output}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
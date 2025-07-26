"""
VATEX-EVAL VCS Evaluation Framework

This module provides a comprehensive VCS evaluation framework for the VATEX-EVAL dataset,
computing VCS scores with different chunk sizes and LCT values.

Features:
- Multiple chunk_size and LCT value evaluation
- Multiple n_refs evaluation (1-9 references)
- Correlation analysis with human judgments
- Parallel processing with checkpointing
- Hierarchical result organization

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
    TextProcessor, EmbeddingGenerator, VATEXEvalUtils
)
from utils.core import DECIMAL_PRECISION, DEFAULT_SEGMENTER_FUNCTION


@dataclass
class VCSResult:
    """Container for VCS metrics from a single candidate-reference pair."""
    video_id: str
    candidate: str
    reference: str
    n_refs: int
    metrics: Dict[str, float]  # VCS_C{chunk}_LCT{n} format
    best_score: float
    best_reference: str
    human_score_1: float
    human_score_2: float
    human_score_3: float
    human_avg: float


@dataclass
class CorrelationResult:
    """Container for correlation results for a specific configuration."""
    n_refs: int
    metric_name: str  # VCS_C{chunk}_LCT{n} format
    kendall_tau: float
    spearman_correlation: float
    best_scores: List[float]


class StructuredLogger:
    """Enhanced logging system with JSON format for automated analysis."""
    
    def __init__(self, log_dir: str, experiment_id: str = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
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
        
        main_handler = logging.FileHandler(self.log_dir / f"vatex-{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    def log_experiment_start(self, lct_value: int, n_refs: int, total_candidates: int):
        """Log start of experiment configuration."""
        self.main_logger.info(f"Starting VCS evaluation: LCT={lct_value} with {n_refs} references")
        self.main_logger.info(f"Total candidates to process: {total_candidates}")
    
    def log_experiment_complete(self, lct_value: int, n_refs: int, processing_time: float, 
                               candidates_processed: int, correlation_results: Dict):
        """Log completion of experiment configuration."""
        self.main_logger.info(f"Completed LCT={lct_value} with {n_refs} references")
        self.main_logger.info(f"Processed {candidates_processed} candidates in {processing_time:.2f}s")
        self.main_logger.info(f"VCS(X,X*) correlation results: {correlation_results}")
    
    def log_correlation_results(self, metric_name: str, kendall_tau: float, spearman_corr: float, 
                               lct_value: int, n_refs: int):
        """Log individual correlation results."""
        self.main_logger.info(f"VCS({metric_name}) LCT_{lct_value}_{n_refs}ref - Kendall: {kendall_tau}, Spearman: {spearman_corr}")
    
    def log_progress(self, current: int, total: int, description: str = "Processing", already_processed: int = 0):
        """Log progress information with resume context."""
        if total > 0:
            percentage = (current / total) * 100
            self.main_logger.info(f"{description}: {current}/{total} ({percentage:.1f}%)")
            
            # Show resume context if applicable
            if already_processed > 0:
                newly_processed = current - already_processed
                self.main_logger.info(f"Resume context: {already_processed} already completed, {newly_processed} processed this session")
        else:
            self.main_logger.info(f"{description}: {current}/0 (N/A%)")
        
        # Calculate ETA based on newly processed items only
        if current > 0 and hasattr(self, '_start_time'):
            elapsed = time.time() - self._start_time
            remaining = total - current
            
            if already_processed > 0:
                # Calculate rate based on newly processed items
                newly_processed = current - already_processed
                if newly_processed > 0:
                    rate = elapsed / newly_processed
                    eta = rate * remaining
                    self.main_logger.info(f"Processing rate: {newly_processed/elapsed:.2f} candidates/sec, ETA: {eta/60:.1f} minutes")
                else:
                    self.main_logger.info(f"No new items processed this session yet")
            else:
                # Standard ETA calculation
                rate = elapsed / current
                eta = rate * remaining
                self.main_logger.info(f"Processing rate: {current/elapsed:.2f} candidates/sec, ETA: {eta/60:.1f} minutes")
    
    def log_error(self, message: str, error: Exception, context: Dict = None):
        """Log errors with context."""
        self.main_logger.error(f"{message}: {error}")
        if context:
            self.main_logger.error(f"Context: {context}")
    
    def set_start_time(self):
        """Set the start time for progress tracking."""
        self._start_time = time.time()


class VCSEvaluator:
    """Core VCS evaluation engine."""
    
    def __init__(self, config: Dict, logger: StructuredLogger = None):
        self.config = config
        self.logger = logger
        self.embedding_fn = EmbeddingGenerator.nv_embed_embedding_fn
        
        # Extract VCS parameters
        vcs_config = config['vcs']
        self.lct_values = vcs_config.get('lct_values', [0])
        self.chunk_size = vcs_config.get('chunk_size', 1)
        self.context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
        self.context_window = vcs_config.get('context_window_control', 4.0)
        self.return_all_metrics = vcs_config.get('return_all_metrics', True)
        self.return_internals = vcs_config.get('return_internals', False)
    
    def compute_vcs_metrics(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Compute VCS metrics for all chunk sizes and LCT values.
        
        Args:
            reference: Reference text
            generated: Generated/candidate text
            
        Returns:
            Dictionary of VCS metrics in VCS_C{chunk}_LCT{n} format
        """
        try:
            segmenter_fn = TextProcessor.get_segmenter_function(DEFAULT_SEGMENTER_FUNCTION)
            vcs_metrics = {}
            
            # Get chunk sizes and LCT values from config
            chunk_sizes = self.config['vcs'].get('chunk_size', [1])
            lct_values = self.lct_values
            
            # Compute VCS for each chunk size and LCT combination
            for chunk_size in chunk_sizes:
                for lct in lct_values:
                    try:
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
                            return_all_metrics=self.return_all_metrics,
                            return_internals=self.return_internals
                        )
                        
                        # Store with hierarchical naming: VCS_C{chunk}_LCT{n}
                        metric_name = f"VCS_C{chunk_size}_LCT{lct}"
                        vcs_metrics[metric_name] = vcs_results.get("VCS", 0.0)
                        
                    except Exception as e:
                        if self.logger:
                            self.logger.log_error(f"VCS computation failed for chunk_size={chunk_size}, LCT={lct}", e)
                        metric_name = f"VCS_C{chunk_size}_LCT{lct}"
                        vcs_metrics[metric_name] = 0.0
            
            return vcs_metrics
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error computing VCS metrics", e)
            
            # Return zero metrics on failure
            zero_metrics = {}
            chunk_sizes = self.config['vcs'].get('chunk_size', [1])
            for chunk_size in chunk_sizes:
                for lct in self.lct_values:
                    metric_name = f"VCS_C{chunk_size}_LCT{lct}"
                    zero_metrics[metric_name] = 0.0
            return zero_metrics
    
    def process_candidate_batch(
        self, 
        candidates: List[str], 
        references_batch: List[List[str]], 
        video_ids: List[str],
        human_scores: np.ndarray,
        n_refs: int,
        start_idx: int = 0
    ) -> List[VCSResult]:
        """Process a batch of candidates for VCS evaluation."""
        
        results = []
        
        for i, (candidate, refs, video_id, human_score_row) in enumerate(zip(candidates, references_batch, video_ids, human_scores)):
            # For each reference, calculate all VCS metrics
            ref_results = []
            
            for ref in refs:
                vcs_metrics = self.compute_vcs_metrics(
                    reference=ref,
                    generated=candidate
                )
                ref_results.append(vcs_metrics)
            
            # For each metric, find best score across references (max aggregation)
            best_metrics = {}
            
            if ref_results:
                # Get all metric names from first result
                metric_names = list(ref_results[0].keys())
                
                for metric_name in metric_names:
                    metric_scores = [ref_result[metric_name] for ref_result in ref_results]
                    
                    # Best score across all references
                    best_metrics[metric_name] = max(metric_scores) if metric_scores else 0.0
                
                # Find best reference for this candidate
                # Use first VCS metric to determine best reference (could be any VCS variant)
                first_vcs_metric = metric_names[0] if metric_names else ""
                if first_vcs_metric:
                    vcs_scores = [ref_result.get(first_vcs_metric, 0.0) for ref_result in ref_results]
                    best_ref_idx = vcs_scores.index(max(vcs_scores)) if vcs_scores else 0
                    best_ref = refs[best_ref_idx] if refs else ""
                    best_score = vcs_scores[best_ref_idx] if vcs_scores else 0.0
                else:
                    best_ref = refs[0] if refs else ""
                    best_score = 0.0
                
                # Extract human scores (3 annotators per candidate)
                human_score_1 = float(human_score_row[0]) if len(human_score_row) > 0 else 0.0
                human_score_2 = float(human_score_row[1]) if len(human_score_row) > 1 else 0.0
                human_score_3 = float(human_score_row[2]) if len(human_score_row) > 2 else 0.0
                human_avg = float(np.mean(human_score_row)) if len(human_score_row) > 0 else 0.0
                
                result = VCSResult(
                    video_id=video_id,
                    candidate=candidate,
                    reference=best_ref,
                    n_refs=n_refs,
                    metrics=best_metrics,
                    best_score=best_score,
                    best_reference=best_ref,
                    human_score_1=human_score_1,
                    human_score_2=human_score_2,
                    human_score_3=human_score_3,
                    human_avg=human_avg
                )
                results.append(result)
        
        return results


class VCSPipeline:
    """Main VCS evaluation pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Extract paths from config
        paths = config['paths']
        
        # Handle relative paths (script is in scripts/ subdirectory)
        script_dir = Path(__file__).parent.parent
        
        # Output folder
        results_dir = Path(paths['results_dir'])
        self.output_folder = str(results_dir if results_dir.is_absolute() else script_dir / results_dir)
        
        # Logs directory
        logs_dir_path = Path(paths.get('logs_dir', 'logs'))
        logs_dir = str(logs_dir_path if logs_dir_path.is_absolute() else script_dir / logs_dir_path)
        logs_dir = Path(logs_dir)
        
        # Extract processing settings
        processing = config['processing']
        max_workers = processing.get('max_workers', 4)
        self.resume_from_checkpoint = processing.get('resume_from_checkpoint', True)
        
        # Create output directories
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
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
        
        # Generate configuration hash
        import hashlib
        config_copy = config.copy()
        config_copy.pop('experiment', None)
        config_str = json.dumps(config_copy, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Initialize components
        self.logger = StructuredLogger(str(logs_dir), self.experiment_id)
        self.checkpoint_manager = CheckpointManager(self.output_folder, self.experiment_id, config_hash)
        self.evaluator = VCSEvaluator(config, self.logger)
        
        # Initialize models
        self.logger.main_logger.info("Initializing models...")
        ModelInitializer.initialize_nvembed(config)
        self.logger.main_logger.info("Models initialized successfully")
    
    def run_vcs_evaluation(self) -> None:
        """Execute the complete VCS evaluation pipeline."""
        
        self.logger.main_logger.info(f"Starting VCS evaluation pipeline")
        self.logger.main_logger.info(f"Output folder: {self.output_folder}")
        self.logger.main_logger.info(f"Experiment ID: {self.experiment_id}")
        
        start_time = time.time()
        
        try:
            # Load VATEX-EVAL data
            vatex_config = self.config['vatex_eval']
            data_dir = vatex_config['data_dir']
            
            self.logger.main_logger.info(f"Loading VATEX-EVAL data from: {data_dir}")
            data = VATEXEvalUtils.load_vatex_data(data_dir)
            
            # Extract candidates, references, and human scores
            cands_all, refs_all, human_scores_array, video_ids_all = VATEXEvalUtils.extract_candidates_and_references(data)
            
            self.logger.main_logger.info(f"Loaded {len(cands_all)} candidates from {len(data)} videos")
            
            # Get evaluation configurations
            use_n_refs_list = vatex_config['use_n_refs']
            
            # Process each n_refs configuration
            all_correlation_results = []
            all_nref_results = {}  # Store results by n_refs for aggregated correlation
            
            for n_refs in use_n_refs_list:
                self.logger.main_logger.info(f"Processing {n_refs} references configuration")
                
                # Set the current configuration for this n_refs
                self.checkpoint_manager.set_current_config(n_refs)
                
                # Check for checkpoint resume with improved logic
                checkpoint_key = f"nrefs_{n_refs}"
                resume_data = None
                if self.resume_from_checkpoint:
                    checkpoint_data = self.checkpoint_manager.load_checkpoint(self.config)
                    if checkpoint_data and checkpoint_data.get('current_config') == checkpoint_key:
                        # Check if this n_refs configuration was fully completed
                        processed_candidates = checkpoint_data.get('processed_candidates', 0)
                        total_candidates = len(cands_all)
                        
                        if processed_candidates >= total_candidates:
                            self.logger.main_logger.info(f"Configuration {checkpoint_key} already completed ({processed_candidates}/{total_candidates})")
                            continue
                        else:
                            self.logger.main_logger.info(f"Resuming {checkpoint_key} from {processed_candidates}/{total_candidates} candidates")
                            resume_data = checkpoint_data
                
                # Limit references
                refs_limited = VATEXEvalUtils.limit_references(refs_all, n_refs)
                
                # Process all candidates with checkpoint saving
                experiment_start_time = time.time()
                results = self._process_with_checkpointing(
                    candidates=cands_all,
                    references_batch=refs_limited,
                    video_ids=video_ids_all,
                    human_scores=human_scores_array,
                    n_refs=n_refs,
                    checkpoint_key=checkpoint_key,
                    resume_data=resume_data
                )
                
                # Compute correlations for this n_refs
                correlation_results = self._compute_correlations(results, human_scores_array, n_refs)
                all_correlation_results.extend(correlation_results)
                all_nref_results[n_refs] = correlation_results
                
                # Save individual results for this n_refs
                self._save_individual_results(results, n_refs)
                
                # Save checkpoint after successful completion (following original ablation pattern)
                if self.resume_from_checkpoint:
                    self.checkpoint_manager.save_checkpoint(
                        processed_candidates=len(results),
                        total_candidates=len(cands_all),
                        current_segmenter=DEFAULT_SEGMENTER_FUNCTION,
                        current_n_refs=n_refs,
                        processing_stats={
                            'current_config': checkpoint_key,
                            'n_refs_value': n_refs
                        }
                    )
                    
                experiment_time = time.time() - experiment_start_time
                avg_correlations = {cr.metric_name: cr.spearman_correlation for cr in correlation_results}
                
                self.logger.main_logger.info(f"Completed {n_refs} references in {experiment_time:.2f}s")
                self.logger.main_logger.info(f"Processed {len(results)} candidates")
                self.logger.main_logger.info(f"Average correlations: {avg_correlations}")
            
            # Save aggregated correlation summary across all n_refs
            if all_nref_results:
                self._save_aggregated_correlation_summary(all_nref_results)
            
            total_time = time.time() - start_time
            self.logger.main_logger.info(f"VCS evaluation completed in {total_time:.2f} seconds")
            
            # Clean up checkpoint files after successful completion
            if self.resume_from_checkpoint:
                self.checkpoint_manager.clear_checkpoint()
            
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise
    
    def _process_with_checkpointing(self, candidates: List[str], references_batch: List[List[str]], 
                                   video_ids: List[str], human_scores: np.ndarray, n_refs: int, 
                                   checkpoint_key: str, resume_data: Dict = None) -> List[VCSResult]:
        """Process candidates with checkpoint saving during processing."""
        
        total_candidates = len(candidates)
        checkpoint_interval = self.config['processing'].get('checkpoint_interval', 100)
        
        # Log the checkpoint interval being used
        self.logger.main_logger.info(f"Using checkpoint interval: {checkpoint_interval} candidates")
        self.logger.main_logger.info(f"Using configuration-specific temp directory: {self.checkpoint_manager.results_dir}")
        
        # Check for existing results and determine starting position
        existing_files = self.checkpoint_manager.get_existing_result_files(n_refs)
        start_idx = 0
        results = []
        processed_count = 0
        last_checkpoint_count = 0
        
        # Handle resume logic - check for existing files regardless of checkpoint
        actual_processed = self.checkpoint_manager.count_processed_candidates_from_files(existing_files, video_ids)
        
        if resume_data:
            # Get resume position from checkpoint
            checkpoint_processed = resume_data.get('processed_candidates', 0)
            batch_start_idx = resume_data.get('processing_stats', {}).get('batch_start_idx', 0)
            
            # Use the minimum of checkpoint and actual files to be safe
            processed_count = min(checkpoint_processed, actual_processed)
            start_idx = processed_count
            last_checkpoint_count = processed_count
            
            self.logger.main_logger.info(f"Resume from checkpoint: checkpoint={checkpoint_processed}, files={actual_processed}, starting from index={start_idx}")
        elif actual_processed > 0:
            # Resume from existing files even without checkpoint
            processed_count = actual_processed
            start_idx = processed_count
            last_checkpoint_count = processed_count
            
            self.logger.main_logger.info(f"Resume from existing files: found {actual_processed} processed files, starting from index={start_idx}")
        
        # Load existing results from temp files if any exist
        if processed_count > 0:
            results = self._load_existing_results_from_temp(existing_files, video_ids[:processed_count], 
                                                           candidates[:processed_count], n_refs)
            self.logger.main_logger.info(f"Loaded {len(results)} existing results from temp files")
        
        self.logger.set_start_time()
        
        # Log initial progress if resuming
        if start_idx > 0:
            self.logger.log_progress(start_idx, total_candidates, f"Resuming {checkpoint_key}", 
                                   already_processed=start_idx)
        
        # Process candidates one by one or in small batches, but save at checkpoint intervals
        batch_size = 1  # Process one candidate at a time for more granular control
        
        for i in range(start_idx, total_candidates, batch_size):
            end_idx = min(i + batch_size, total_candidates)
            batch_candidates = candidates[i:end_idx]
            batch_references = references_batch[i:end_idx]
            batch_video_ids = video_ids[i:end_idx]
            batch_human_scores = human_scores[i:end_idx]
            
            # Process batch
            batch_results = self.evaluator.process_candidate_batch(
                candidates=batch_candidates,
                references_batch=batch_references,
                video_ids=batch_video_ids,
                human_scores=batch_human_scores,
                n_refs=n_refs,
                start_idx=i
            )
            
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            # Check if we should save checkpoint
            candidates_since_last_checkpoint = processed_count - last_checkpoint_count
            if self.resume_from_checkpoint and candidates_since_last_checkpoint >= checkpoint_interval:
                # Save recent results to temporary directory
                recent_results = results[last_checkpoint_count:processed_count]
                self._save_batch_results(recent_results, processed_count, n_refs)
                
                self.logger.main_logger.info(f"Saving checkpoint after {candidates_since_last_checkpoint} new candidates (total: {processed_count})")
                
                self.checkpoint_manager.save_checkpoint(
                    processed_candidates=processed_count,
                    total_candidates=total_candidates,
                    current_segmenter=DEFAULT_SEGMENTER_FUNCTION,
                    current_n_refs=n_refs,
                    processing_stats={
                        'current_config': checkpoint_key,
                        'n_refs_value': n_refs,
                        'batch_start_idx': i,
                        'batch_end_idx': end_idx,
                        'avg_time_per_candidate': (time.time() - self.logger._start_time) / processed_count if processed_count > 0 else 1.0
                    }
                )
                last_checkpoint_count = processed_count  # Update last checkpoint count
            
            # Log progress with resume context
            if processed_count % 50 == 0 or processed_count == total_candidates:
                self.logger.log_progress(processed_count, total_candidates, f"Processing {checkpoint_key}", 
                                       already_processed=start_idx if resume_data else 0)
        
        # Save any remaining results at the end (if not already saved)
        if self.resume_from_checkpoint and processed_count > last_checkpoint_count:
            # Save the final results that weren't saved at a checkpoint interval
            final_results = results[last_checkpoint_count:]
            if final_results:
                self.logger.main_logger.info(f"Saving final {len(final_results)} results that weren't checkpointed")
                self._save_batch_results(final_results, processed_count, n_refs)
        
        return results
    
    def _load_existing_results_from_temp(self, existing_files: set, video_ids: List[str], 
                                        candidates: List[str], n_refs: int) -> List[VCSResult]:
        """Load existing results from temporary JSON files for the current n_refs configuration."""
        results = []
        
        # Group video_ids and candidates by video_id
        video_to_candidates = {}
        for i, (video_id, candidate) in enumerate(zip(video_ids, candidates)):
            if video_id not in video_to_candidates:
                video_to_candidates[video_id] = []
            video_to_candidates[video_id].append((i, candidate))
        
        # Load results from existing files (configuration-specific)
        config_results_dir = self.checkpoint_manager.base_results_dir / f"nrefs_{n_refs}"
        for video_id in existing_files:
            if video_id in video_to_candidates:
                json_file = config_results_dir / f"{video_id}.json"
                
                try:
                    with open(json_file, 'r') as f:
                        video_data = json.load(f)
                    
                    # Convert JSON data back to VCSResult objects
                    for result_dict in video_data:
                        result = VCSResult(
                            video_id=result_dict['video_id'],
                            candidate=result_dict['candidate'],
                            reference=result_dict['reference'],
                            n_refs=result_dict['n_refs'],
                            metrics=result_dict['metrics'],
                            best_score=result_dict['best_score'],
                            best_reference=result_dict['best_reference'],
                            human_score_1=result_dict['human_score_1'],
                            human_score_2=result_dict['human_score_2'],
                            human_score_3=result_dict['human_score_3'],
                            human_avg=result_dict['human_avg']
                        )
                        results.append(result)
                        
                except Exception as e:
                    if self.logger:
                        self.logger.log_error(f"Failed to load existing results from {json_file}", e)
        
        # Sort results to maintain order (by video_id and then by candidate order)
        results.sort(key=lambda r: (video_ids.index(r.video_id) if r.video_id in video_ids else 0))
        
        return results
    
    def _save_batch_results(self, batch_results: List[VCSResult], processed_count: int, n_refs: int) -> None:
        """Save batch results incrementally to both temporary and individual results directories."""
        
        # Get the temporary results directory from checkpoint manager
        temp_results_dir = self.checkpoint_manager.results_dir
        
        # Also get the individual results directory
        individual_results_dir = Path(self.output_folder) / "individual_results" / f"{n_refs}ref"
        individual_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Group results by video_id (same as individual_results structure)
        video_groups = {}
        for result in batch_results:
            video_id = result.video_id
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(result)
        
        # Save individual files per video_id to both temp and individual results directories
        files_saved = 0
        decimal_precision = self.config['output'].get('decimal_precision', DECIMAL_PRECISION)
        
        for video_id, video_results in video_groups.items():
            # Convert video results to serializable format for temp JSON
            video_data_json = []
            video_data_csv = []
            
            for candidate_idx, result in enumerate(video_results, 1):
                # JSON format for temp directory (full data)
                result_dict = {
                    'candidate_id': candidate_idx,
                    'video_id': result.video_id,
                    'candidate': result.candidate,
                    'reference': result.reference,
                    'n_refs': result.n_refs,
                    'metrics': result.metrics,
                    'best_score': result.best_score,
                    'best_reference': result.best_reference,
                    'human_score_1': result.human_score_1,
                    'human_score_2': result.human_score_2,
                    'human_score_3': result.human_score_3,
                    'human_avg': result.human_avg
                }
                video_data_json.append(result_dict)
                
                # CSV format for individual results directory (formatted like final output)
                csv_row = {
                    'candidate_id': candidate_idx,
                    'candidate': result.candidate,
                    'best_score': round(result.best_score, decimal_precision),
                    'best_reference': result.best_reference,
                    'human_score_1': round(result.human_score_1, decimal_precision),
                    'human_score_2': round(result.human_score_2, decimal_precision),
                    'human_score_3': round(result.human_score_3, decimal_precision),
                    'human_avg': round(result.human_avg, decimal_precision),
                    **{k: round(v, decimal_precision) for k, v in result.metrics.items()}
                }
                video_data_csv.append(csv_row)
            
            # Save to temp directory as JSON
            video_json_path = temp_results_dir / f"{video_id}.json"
            try:
                with open(video_json_path, 'w') as f:
                    json.dump(video_data_json, f, indent=2)
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Failed to save temp JSON for {video_id}", e)
                continue
            
            # Save to individual results directory as CSV
            video_csv_path = individual_results_dir / f"{video_id}.csv"
            try:
                import pandas as pd
                video_df = pd.DataFrame(video_data_csv)
                video_df.to_csv(video_csv_path, index=False)
                files_saved += 1
            except Exception as e:
                if self.logger:
                    self.logger.log_error(f"Failed to save individual CSV for {video_id}", e)
        
        if self.logger and files_saved > 0:
            self.logger.main_logger.info(f"Saved {files_saved} video files to both temp directory (JSON) and individual results directory (CSV)")
    
    def _compute_correlations(self, results: List[VCSResult], human_scores: np.ndarray, n_refs: int) -> List[CorrelationResult]:
        """Compute correlations between metrics and human judgments."""
        
        correlation_results = []
        
        if not results:
            return correlation_results
        
        # Get all metric names
        metric_names = list(results[0].metrics.keys())
        
        for metric_name in metric_names:
            # Extract scores for this metric
            metric_scores = [result.metrics[metric_name] for result in results]
            
            # Print correlation section header
            print(f'VCS({metric_name}) correlation {n_refs}ref ------------------------------')
            
            # Compute correlation
            kendall_tau, spearman_corr = VATEXEvalUtils.compute_correlation_uniquehuman(
                np.array(metric_scores), human_scores
            )
            
            # Log correlation results to file
            if self.logger:
                self.logger.main_logger.info(f"VCS({metric_name}) {n_refs}ref - Kendall: {kendall_tau:.4f}, Spearman: {spearman_corr:.4f}")
            
            correlation_result = CorrelationResult(
                n_refs=n_refs,
                metric_name=metric_name,
                kendall_tau=kendall_tau,
                spearman_correlation=spearman_corr,
                best_scores=metric_scores
            )
            correlation_results.append(correlation_result)
        
        return correlation_results
    
    def _save_individual_results(self, results: List[VCSResult], n_refs: int) -> None:
        """Save individual results for a specific n_refs configuration."""
        
        # Create output directory with new structure: results/vatex_eval/individual_results/Nref/
        individual_results_dir = Path(self.output_folder) / "individual_results" / f"{n_refs}ref"
        os.makedirs(individual_results_dir, exist_ok=True)
        
        # Save detailed scores (individual CSV files by video_id)
        if self.config['output'].get('save_detailed_scores', True):
            self._save_detailed_scores(results, str(individual_results_dir), n_refs)
    
    def _save_detailed_scores(self, results: List[VCSResult], 
                             individual_results_dir: str, n_refs: int) -> None:
        """Save detailed scores for individual candidates."""
        
        decimal_precision = self.config['output'].get('decimal_precision', DECIMAL_PRECISION)
        
        # Group results by video_id
        video_groups = {}
        for result in results:
            video_id = result.video_id
            if video_id not in video_groups:
                video_groups[video_id] = []
            video_groups[video_id].append(result)
        
        # Save individual CSV files for each video_id
        individual_files_saved = 0
        for video_id, video_results in video_groups.items():
            video_data = []
            for candidate_idx, result in enumerate(video_results, 1):
                row = {
                    'candidate_id': candidate_idx,
                    'candidate': result.candidate,
                    'best_score': round(result.best_score, decimal_precision),
                    'best_reference': result.best_reference,
                    'human_score_1': round(result.human_score_1, decimal_precision),
                    'human_score_2': round(result.human_score_2, decimal_precision),
                    'human_score_3': round(result.human_score_3, decimal_precision),
                    'human_avg': round(result.human_avg, decimal_precision),
                    **{k: round(v, decimal_precision) for k, v in result.metrics.items()}
                }
                video_data.append(row)
            
            # Save individual video CSV
            video_df = pd.DataFrame(video_data)
            video_output_path = Path(individual_results_dir) / f"{video_id}.csv"
            video_df.to_csv(video_output_path, index=False)
            individual_files_saved += 1
        
        # Note: No longer saving aggregated detailed_scores.csv as individual files contain all data
        
        self.logger.main_logger.info(f"Saved {individual_files_saved} individual video CSV files to: {individual_results_dir}")
    
    def _save_aggregated_correlation_summary(self, all_nref_results: Dict[int, List[CorrelationResult]]) -> None:
        """Save hierarchical correlation summary across all n_refs configurations."""
        
        # Create aggregated results directory
        aggregated_results_dir = Path(self.output_folder) / "aggregated_results"
        os.makedirs(aggregated_results_dir, exist_ok=True)
        
        # Collect all unique metrics across all n_refs
        all_metrics = set()
        for correlation_results in all_nref_results.values():
            for result in correlation_results:
                all_metrics.add(result.metric_name)
        
        # Sort metrics for consistent ordering
        sorted_metrics = sorted(all_metrics)
        
        # Build hierarchical DataFrame
        data = {"Metric": sorted_metrics}
        
        # Add columns for each n_refs configuration
        for n_refs in sorted(all_nref_results.keys()):
            correlation_results = all_nref_results[n_refs]
            
            # Create mapping from metric to correlation values
            metric_to_kendall = {}
            metric_to_spearman = {}
            
            for result in correlation_results:
                metric_to_kendall[result.metric_name] = result.kendall_tau
                metric_to_spearman[result.metric_name] = result.spearman_correlation
            
            # Add columns with hierarchical naming
            kendall_col = f"{n_refs}ref_Kendall_Tau"
            spearman_col = f"{n_refs}ref_Spearman_Correlation"
            
            data[kendall_col] = [metric_to_kendall.get(metric, 0.0) for metric in sorted_metrics]
            data[spearman_col] = [metric_to_spearman.get(metric, 0.0) for metric in sorted_metrics]
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        output_path = aggregated_results_dir / "correlation_summary.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.main_logger.info(f"Saved aggregated correlation summary to: {output_path}")
    


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VATEX-EVAL VCS Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/vatex-eval.py --config config/vatex-eval.yaml
  python scripts/vatex-eval.py --config config/vatex-eval.yaml --verbose

For detailed configuration options, see config/vatex-eval.yaml file.
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
    """Main execution function."""
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
        pipeline = VCSPipeline(config)
        pipeline.run_vcs_evaluation()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
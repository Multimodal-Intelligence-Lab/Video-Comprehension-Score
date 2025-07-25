"""
VATEX-EVAL Ablation Study Framework

This module provides a comprehensive ablation study framework for VCS metrics
on the VATEX-EVAL dataset, following the MPII pipeline structure.

Features:
- Multiple segmenter function evaluation
- Multiple n_refs evaluation (1-9 references)
- Correlation analysis with human judgments
- Parallel processing with checkpointing
- Structured logging and result organization

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
from utils.core import ABLATION_METRICS_ORDER, DECIMAL_PRECISION, DEFAULT_SEGMENTER_FUNCTION


@dataclass
class AblationResult:
    """Container for ablation metrics from a single candidate-reference pair."""
    video_id: str
    candidate: str
    reference: str
    n_refs: int
    lct_value: int
    metrics: Dict[str, float]
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
    lct_value: int
    metric_name: str
    kendall_tau: float
    spearman_correlation: float
    best_scores: List[float]


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
        
        main_handler = logging.FileHandler(self.log_dir / f"vatex_{self.experiment_id}.log")
        main_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        main_handler.setFormatter(main_formatter)
        self.main_logger.addHandler(main_handler)
    
    def log_experiment_start(self, lct_value: int, n_refs: int, total_candidates: int):
        """Log start of experiment configuration."""
        self.main_logger.info(f"Starting ablation experiment: LCT={lct_value} with {n_refs} references")
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
    
    def log_progress(self, current: int, total: int, description: str = "Processing"):
        """Log progress information."""
        if total > 0:
            self.main_logger.info(f"{description}: {current}/{total} ({current/total*100:.1f}%)")
        else:
            self.main_logger.info(f"{description}: {current}/0 (N/A%)")
        
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
        self.lct_values = vcs_config.get('lct_values', [0])
        self.chunk_size = vcs_config.get('chunk_size', 1)
        self.context_cutoff = vcs_config.get('context_cutoff_value', 0.6)
        self.context_window = vcs_config.get('context_window_control', 4.0)
        self.return_all_metrics = vcs_config.get('return_all_metrics', True)
        self.return_internals = vcs_config.get('return_internals', False)
    
    def compute_ablation_metrics(self, reference: str, generated: str, lct_value: int) -> Dict[str, float]:
        """
        Compute all ablation metrics for a single candidate-reference pair.
        
        Args:
            reference: Reference text
            generated: Generated/candidate text
            lct_value: LCT value to use for computation
            
        Returns:
            Dictionary of ablation metrics without LCT suffixes
        """
        try:
            segmenter_fn = TextProcessor.get_segmenter_function(DEFAULT_SEGMENTER_FUNCTION)
            
            vcs_results = vcs.compute_vcs_score(
                reference_text=reference,
                generated_text=generated,
                segmenter_fn=segmenter_fn,
                embedding_fn_las=self.embedding_fn,
                embedding_fn_gas=self.embedding_fn,
                chunk_size=self.chunk_size,
                context_cutoff_value=self.context_cutoff,
                context_window_control=self.context_window,
                lct=lct_value,
                return_all_metrics=self.return_all_metrics,
                return_internals=self.return_internals
            )
            
            # Extract base metrics
            SAS = vcs_results.get("GAS", 0.0)
            CAS = vcs_results.get("LAS", 0.0)
            NAS_D = vcs_results.get("NAS-D", 0.0)
            NAS_L = vcs_results.get("NAS-L", 0.0)
            NAS = vcs_results.get("NAS", 0.0)
            sas_cas_scaled = vcs_results.get("GAS-LAS-Scaled", 0.0)
            vad_scaled = vcs_results.get("VCS", 0.0)
            
            # Helper functions for metric computation
            def compute_sas_cas_scaled(SAS, CAS):
                if CAS <= 0:
                    return 0.0
                val = SAS - (1 - CAS)
                return (val / CAS) if (val > 0) else 0.0
            
            def compute_sas_nas_scaled(SAS, NAS):
                if SAS < NAS:
                    numerator = SAS - (1 - NAS)
                    denominator = NAS
                else:
                    numerator = NAS - (1 - SAS)
                    denominator = SAS
                return (numerator / denominator) if (numerator > 0 and denominator != 0) else 0.0
            
            # Compute additional metrics
            nas_plus_cas_scaled = compute_sas_cas_scaled(NAS, CAS)
            sas_plus_nas_l_scaled = compute_sas_nas_scaled(SAS, NAS_L)
            sas_plus_nas_d_scaled = compute_sas_nas_scaled(SAS, NAS_D)
            sas_plus_nas_scaled = compute_sas_nas_scaled(SAS, NAS)
            sas_cas_s_plus_nas_d = compute_sas_nas_scaled(sas_cas_scaled, NAS_D)
            sas_cas_s_plus_nas_l = compute_sas_nas_scaled(sas_cas_scaled, NAS_L)
            
            # Build ablation table without LCT suffixes
            ablation_table = {
                "GAS": SAS,
                "LAS": CAS,
                "NAS-D": NAS_D,
                "NAS-L": NAS_L,
                "NAS": NAS,
                "NAS\n+LAS(S)": nas_plus_cas_scaled,
                "GAS\n+LAS(S)": sas_cas_scaled,
                "GAS\n+NAS-L(S)": sas_plus_nas_l_scaled,
                "GAS\n+NAS-D(S)": sas_plus_nas_d_scaled,
                "GAS\n+NAS(S)": sas_plus_nas_scaled,
                "GAS\n+LAS(S)\n+NAS-D(S)": sas_cas_s_plus_nas_d,
                "GAS\n+LAS(S)\n+NAS-L(S)": sas_cas_s_plus_nas_l,
                "GAS\n+LAS(S)\n+(NAS-D\n+NAS-L)(S)": vad_scaled,
            }
            
            return ablation_table
            
        except Exception as e:
            if self.logger:
                self.logger.log_error(f"Error computing ablation metrics for LCT={lct_value}", e)
            
            # Return zero metrics on failure
            zero_metrics = {}
            for base_metric in ABLATION_METRICS_ORDER:
                zero_metrics[base_metric] = 0.0
            return zero_metrics
    
    def process_candidate_batch(
        self, 
        candidates: List[str], 
        references_batch: List[List[str]], 
        video_ids: List[str],
        human_scores: np.ndarray,
        lct_value: int,
        n_refs: int,
        start_idx: int = 0
    ) -> List[AblationResult]:
        """Process a batch of candidates for ablation evaluation."""
        
        results = []
        
        for i, (candidate, refs, video_id, human_score_row) in enumerate(zip(candidates, references_batch, video_ids, human_scores)):
            # For each reference, calculate all ablation metrics
            ref_results = []
            
            for ref in refs:
                ablation_metrics = self.compute_ablation_metrics(
                    reference=ref,
                    generated=candidate,
                    lct_value=lct_value
                )
                ref_results.append(ablation_metrics)
            
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
                # Use VCS score to determine best reference
                vcs_key = "GAS\n+LAS(S)\n+(NAS-D\n+NAS-L)(S)"
                vcs_scores = [ref_result.get(vcs_key, 0.0) for ref_result in ref_results]
                best_ref_idx = vcs_scores.index(max(vcs_scores)) if vcs_scores else 0
                best_ref = refs[best_ref_idx] if refs else ""
                best_score = vcs_scores[best_ref_idx] if vcs_scores else 0.0
                
                # Extract human scores (3 annotators per candidate)
                human_score_1 = float(human_score_row[0]) if len(human_score_row) > 0 else 0.0
                human_score_2 = float(human_score_row[1]) if len(human_score_row) > 1 else 0.0
                human_score_3 = float(human_score_row[2]) if len(human_score_row) > 2 else 0.0
                human_avg = float(np.mean(human_score_row)) if len(human_score_row) > 0 else 0.0
                
                result = AblationResult(
                    video_id=video_id,
                    candidate=candidate,
                    reference=best_ref,
                    n_refs=n_refs,
                    lct_value=lct_value,
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


class AblationPipeline:
    """Main ablation evaluation pipeline."""
    
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
        ablation_logs_dir = Path(logs_dir)
        
        # Extract processing settings
        processing = config['processing']
        max_workers = processing.get('max_workers', 4)
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
        self.logger = StructuredLogger(str(ablation_logs_dir), self.experiment_id)
        self.checkpoint_manager = CheckpointManager(self.output_folder, self.experiment_id, config_hash)
        self.evaluator = AblationEvaluator(config, self.logger)
        
        # Initialize models
        self.logger.main_logger.info("Initializing models...")
        ModelInitializer.initialize_nvembed(config)
        self.logger.main_logger.info("Models initialized successfully")
    
    def run_ablation_study(self) -> None:
        """Execute the complete ablation study pipeline."""
        
        self.logger.main_logger.info(f"Starting ablation study pipeline")
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
            
            # Process each configuration
            all_correlation_results = []
            
            total_configurations = len(self.evaluator.lct_values) * len(use_n_refs_list)
            config_count = 0
            
            for lct_value in self.evaluator.lct_values:
                for n_refs in use_n_refs_list:
                    config_count += 1
                    self.logger.log_experiment_start(lct_value, n_refs, len(cands_all))
                    
                    # Check for checkpoint resume
                    checkpoint_key = f"LCT_{lct_value}_nrefs_{n_refs}"
                    if self.resume_from_checkpoint:
                        checkpoint_data = self.checkpoint_manager.load_checkpoint(self.config)
                        if checkpoint_data and checkpoint_data.get('current_config') == checkpoint_key:
                            self.logger.main_logger.info(f"Resuming from checkpoint for {checkpoint_key}")
                            continue
                    
                    # Limit references
                    refs_limited = VATEXEvalUtils.limit_references(refs_all, n_refs)
                    
                    # Process all candidates with checkpoint saving
                    experiment_start_time = time.time()
                    results = self._process_with_checkpointing(
                        candidates=cands_all,
                        references_batch=refs_limited,
                        video_ids=video_ids_all,
                        human_scores=human_scores_array,
                        lct_value=lct_value,
                        n_refs=n_refs,
                        checkpoint_key=checkpoint_key
                    )
                    
                    # Compute correlations
                    correlation_results = self._compute_correlations(results, human_scores_array)
                    all_correlation_results.extend(correlation_results)
                    
                    # Save results
                    self._save_experiment_results(results, correlation_results, lct_value, n_refs)
                    
                    # Save checkpoint after successful completion
                    if self.resume_from_checkpoint:
                        self.checkpoint_manager.save_checkpoint(
                            processed_candidates=len(results),
                            total_candidates=len(cands_all),
                            current_segmenter=DEFAULT_SEGMENTER_FUNCTION,
                            current_n_refs=n_refs,
                            processing_stats={
                                'config_count': config_count,
                                'total_configurations': total_configurations,
                                'current_config': checkpoint_key,
                                'lct_value': lct_value
                            }
                        )
                    
                    experiment_time = time.time() - experiment_start_time
                    avg_correlations = {cr.metric_name: cr.spearman_correlation for cr in correlation_results}
                    
                    self.logger.log_experiment_complete(
                        lct_value, n_refs, experiment_time, 
                        len(results), avg_correlations
                    )
            
            # Note: No longer saving comprehensive results as they are not needed
            
            total_time = time.time() - start_time
            self.logger.main_logger.info(f"Ablation study completed in {total_time:.2f} seconds")
            
            # Clean up checkpoint files after successful completion
            if self.resume_from_checkpoint:
                self.checkpoint_manager.clear_checkpoint()
            
        except Exception as e:
            self.logger.log_error("Pipeline execution failed", e)
            raise
    
    def _process_with_checkpointing(self, candidates: List[str], references_batch: List[List[str]], 
                                   video_ids: List[str], human_scores: np.ndarray, lct_value: int, n_refs: int, 
                                   checkpoint_key: str) -> List[AblationResult]:
        """Process candidates with checkpoint saving during processing."""
        
        total_candidates = len(candidates)
        checkpoint_interval = self.config['processing'].get('checkpoint_interval', 100)
        
        # Process candidates in batches with checkpointing
        results = []
        processed_count = 0
        
        self.logger.set_start_time()
        
        for i in range(0, total_candidates, checkpoint_interval):
            end_idx = min(i + checkpoint_interval, total_candidates)
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
                lct_value=lct_value,
                n_refs=n_refs,
                start_idx=i
            )
            
            results.extend(batch_results)
            processed_count += len(batch_results)
            
            # Save checkpoint
            if self.resume_from_checkpoint and self.checkpoint_manager.should_save_checkpoint(len(batch_results)):
                self.checkpoint_manager.save_checkpoint(
                    processed_candidates=processed_count,
                    total_candidates=total_candidates,
                    current_segmenter=DEFAULT_SEGMENTER_FUNCTION,
                    current_n_refs=n_refs,
                    processing_stats={
                        'current_config': checkpoint_key,
                        'lct_value': lct_value,
                        'batch_start_idx': i,
                        'batch_end_idx': end_idx,
                        'avg_time_per_candidate': (time.time() - self.logger._start_time) / processed_count if processed_count > 0 else 1.0
                    }
                )
            
            # Log progress
            if processed_count % 50 == 0 or processed_count == total_candidates:
                self.logger.log_progress(processed_count, total_candidates, f"Processing {checkpoint_key}")
        
        return results
    
    def _compute_correlations(self, results: List[AblationResult], human_scores: np.ndarray) -> List[CorrelationResult]:
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
            print(f'VCS({metric_name}) correlation LCT_{results[0].lct_value}_{results[0].n_refs}ref ------------------------------')
            
            # Compute correlation
            kendall_tau, spearman_corr = VATEXEvalUtils.compute_correlation_uniquehuman(
                np.array(metric_scores), human_scores
            )
            
            # Log correlation results to file
            if self.logger:
                self.logger.log_correlation_results(metric_name, kendall_tau, spearman_corr, 
                                                   results[0].lct_value, results[0].n_refs)
            
            correlation_result = CorrelationResult(
                n_refs=results[0].n_refs,
                lct_value=results[0].lct_value,
                metric_name=metric_name,
                kendall_tau=kendall_tau,
                spearman_correlation=spearman_corr,
                best_scores=metric_scores
            )
            correlation_results.append(correlation_result)
        
        return correlation_results
    
    def _save_experiment_results(self, results: List[AblationResult], 
                                correlation_results: List[CorrelationResult],
                                lct_value: int, n_refs: int) -> None:
        """Save results for a single experiment configuration."""
        
        # Create output directory with new structure: results/vatex_eval/ablation/LCT_X/Nref/
        base_results_dir = Path(self.output_folder) / "ablation" / f"LCT_{lct_value}" / f"{n_refs}ref"
        individual_results_dir = base_results_dir / "individual_results"
        aggregated_results_dir = base_results_dir / "aggregated_results"
        
        os.makedirs(individual_results_dir, exist_ok=True)
        os.makedirs(aggregated_results_dir, exist_ok=True)
        
        # Save detailed scores (individual CSV files by video_id)
        if self.config['output'].get('save_detailed_scores', True):
            self._save_detailed_scores(results, str(individual_results_dir), str(aggregated_results_dir), lct_value, n_refs)
        
        # Save correlation summary (in aggregated_results)
        if self.config['output'].get('save_correlation_summary', True):
            self._save_correlation_summary(correlation_results, str(aggregated_results_dir), lct_value, n_refs)
    
    def _save_detailed_scores(self, results: List[AblationResult], 
                             individual_results_dir: str, aggregated_results_dir: str, 
                             lct_value: int, n_refs: int) -> None:
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
    
    def _save_correlation_summary(self, correlation_results: List[CorrelationResult],
                                 results_dir: str, lct_value: int, n_refs: int) -> None:
        """Save correlation summary for this configuration."""
        
        # Create DataFrame
        data = []
        for result in correlation_results:
            row = {
                'Metric': result.metric_name,
                'Kendall_Tau': result.kendall_tau,
                'Spearman_Correlation': result.spearman_correlation
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to CSV with simple filename (LCT info is in folder structure)
        output_path = Path(results_dir) / f"correlation_summary.csv"
        df.to_csv(output_path, index=False)
        
        self.logger.main_logger.info(f"Saved correlation summary to: {output_path}")
    


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VATEX-EVAL Ablation Study Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/vatex_eval_ablation.py --config config/ablation.yaml
  python scripts/vatex_eval_ablation.py --config config/ablation.yaml --verbose

For detailed configuration options, see config/ablation.yaml file.
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
        pipeline = AblationPipeline(config)
        pipeline.run_ablation_study()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    main()
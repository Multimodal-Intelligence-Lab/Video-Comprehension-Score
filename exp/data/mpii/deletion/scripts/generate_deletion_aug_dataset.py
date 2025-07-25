#!/usr/bin/env python3
"""
Data augmentation script for deletion experiments.
Generates augmented datasets by iteratively deleting text segments.

This script processes ground truth video captions and creates multiple iterations
where segments are progressively deleted from different positions:
- beginning: Deleted from the start of the text
- middle: Deleted from the middle position  
- end: Deleted from the end of the text
- random: Deleted from random positions

Author: Generated from addition scripts with deletion-specific adaptations
"""

import os
import re
import sys
import math
import json
import time
import random
import string
import logging
import yaml
import torch
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from wtpsplit import SaT
from typing import List, Dict, Any, Optional
import glob

# ============================================================================
# STANDALONE SAT MODEL AND TEXT PROCESSING
# ============================================================================

class ModelInitializer:
    """Handles initialization of SAT model for sentence segmentation."""
    
    # Class-level storage for SAT model
    _sat_model = None
    
    @classmethod
    def initialize_sat(cls) -> None:
        """Initialize SAT model for sentence segmentation."""
        if cls._sat_model is None:
            print("Initializing SAT model...")
            cls._sat_model = SaT("sat-12l-sm")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cls._sat_model.half().to(device)
            print(f"SAT model initialized on {device}")
    
    @classmethod
    def get_sat_model(cls):
        """Get the initialized SAT model."""
        return cls._sat_model

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
        sat_model = ModelInitializer.get_sat_model()
        
        if sat_model is None:
            raise RuntimeError("SAT model not initialized. Call ModelInitializer.initialize_sat() first.")
        
        # Direct segmentation without preprocessing (matches notebook behavior)
        sentences = sat_model.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def clean_trailing_punctuation(segment: str) -> str:
        """Clean up trailing punctuation marks."""
        segment = segment.strip()
        segment = re.sub(r'[.]+$', '.', segment)
        segment = re.sub(r'[!]+$', '!', segment)
        segment = re.sub(r'[?]+$', '?', segment)
        return segment
    
    @staticmethod
    def join_segments(segments: List[str]) -> str:
        """Join segments with proper punctuation."""
        cleaned = []
        for seg in segments:
            seg = TextProcessor.clean_trailing_punctuation(seg)
            if seg and seg[-1] not in {'.', '!', '?'}:
                seg += '.'
            cleaned.append(seg)
        return ' '.join(cleaned)
    
    @staticmethod
    def limit_to_fixed_segments(original_text: str, num_segments: int) -> str:
        """Truncate text to first num_segments using SAT model."""
        all_segments = TextProcessor.sat_segmenter(original_text)
        if len(all_segments) > num_segments:
            all_segments = all_segments[:num_segments]
        return TextProcessor.join_segments(all_segments)

# ============================================================================
# CONFIGURATION AND LOGGING
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

def setup_logging(config: Dict[str, Any]) -> Optional[logging.Logger]:
    """Setup logging configuration."""
    if not config['processing']['enable_logging']:
        return None
    
    # Create logs directory
    logs_dir = Path(config['output']['logs_dir'])
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"deletion_aug_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def load_ground_truth_files(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load all ground truth JSON files."""
    gt_pattern = config['data']['gt_file_path']
    gt_files = glob.glob(gt_pattern)
    
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found matching: {gt_pattern}")
    
    all_data = {}
    for gt_file in gt_files:
        print(f"Loading: {gt_file}")
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.update(data)
        except Exception as e:
            print(f"Error loading {gt_file}: {e}")
    
    return all_data

# ============================================================================
# DELETION TRANSFORMATIONS
# ============================================================================

class DeletionProcessor:
    """Handles iterative deletion of text segments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.random_seed = config['deletions']['random_seed']
        self.del_segments = config['deletions']['del_segments']
        random.seed(self.random_seed)
    
    def delete_segments_at_beginning(self, text: str, del_count: int = None) -> str:
        """Delete segments from the beginning."""
        if del_count is None:
            del_count = self.del_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        to_delete = min(del_count, len(current_segments) - 1)  # Avoid deleting all
        segments_to_keep = current_segments[to_delete:]
        return TextProcessor.join_segments(segments_to_keep)
    
    def delete_segments_in_middle(self, text: str, del_count: int = None) -> str:
        """Delete segments from the middle."""
        if del_count is None:
            del_count = self.del_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        to_delete = min(del_count, len(current_segments) - 1)
        mid_index = len(current_segments) // 2
        delete_start = mid_index - (to_delete // 2)
        delete_end = delete_start + to_delete
        
        # Stay within bounds
        delete_start = max(0, delete_start)
        delete_end = min(len(current_segments), delete_end)
        
        segments_to_keep = current_segments[:delete_start] + current_segments[delete_end:]
        return TextProcessor.join_segments(segments_to_keep)
    
    def delete_segments_at_end(self, text: str, del_count: int = None) -> str:
        """Delete segments from the end."""
        if del_count is None:
            del_count = self.del_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        to_delete = min(del_count, len(current_segments) - 1)
        segments_to_keep = current_segments[:-to_delete]
        return TextProcessor.join_segments(segments_to_keep)
    
    def delete_segments_randomly(self, text: str, del_count: int = None) -> str:
        """Randomly delete segments from within the existing segments."""
        if del_count is None:
            del_count = self.del_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        to_delete = min(del_count, len(current_segments) - 1)
        # Randomly select 'to_delete' indices to remove
        indices_to_delete = sorted(
            random.sample(range(len(current_segments)), k=to_delete), 
            reverse=True
        )
        segments_to_keep = list(current_segments)
        
        for idx in indices_to_delete:
            segments_to_keep.pop(idx)
        
        return TextProcessor.join_segments(segments_to_keep)
    
    def generate_iterations(self, truncated_text: str, position: str, 
                          num_iterations: int) -> Dict[str, str]:
        """
        Generate multiple iterations with cumulative deletions.
        
        Args:
            truncated_text: Pre-truncated ground truth text
            position: Position type for deletions
            num_iterations: Number of iterations to generate
            
        Returns:
            Dictionary mapping iteration numbers to modified texts
        """
        iterations = {}
        current_text = truncated_text
        
        for i in range(1, num_iterations + 1):
            # Delete segments based on position
            if position == 'beginning':
                current_text = self.delete_segments_at_beginning(current_text)
            elif position == 'middle':
                current_text = self.delete_segments_in_middle(current_text)
            elif position == 'end':
                current_text = self.delete_segments_at_end(current_text)
            elif position == 'random':
                current_text = self.delete_segments_randomly(current_text)
            else:
                raise ValueError(f"Unknown position: {position}")
            
            iterations[str(i)] = current_text
        
        return iterations

# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================

def setup_output_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create output directory structure for deletion experiment."""
    base_output_dir = Path(config['output']['output_dir'])
    positions = config['deletions']['positions']
    
    output_dirs = {}
    for position in positions:
        position_dir = base_output_dir / position
        position_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[position] = position_dir
    
    return output_dirs

def check_existing_output(video_id: str, position: str, output_dirs: Dict[str, Path],
                         num_iterations: int) -> bool:
    """
    Check if output already exists for resume functionality.
    
    Args:
        video_id: Video ID to check
        position: Position type
        output_dirs: Output directory mapping
        num_iterations: Expected number of iterations
        
    Returns:
        True if complete output exists, False otherwise
    """
    output_file = output_dirs[position] / f"{video_id}.json"
    
    if not output_file.exists():
        return False
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if all iterations exist
        iterations = data.get('iterations', {})
        expected_iterations = set(str(i) for i in range(1, num_iterations + 1))
        existing_iterations = set(iterations.keys())
        
        return expected_iterations.issubset(existing_iterations)
    
    except Exception:
        return False

def save_deletion_output(video_id: str, truncated_ground_truth: str, iterations: Dict[str, str],
                        position: str, output_dirs: Dict[str, Path]) -> None:
    """Save deletion iterations to output file."""
    output_file = output_dirs[position] / f"{video_id}.json"
    
    output_data = {
        "ground_truth": truncated_ground_truth,
        "iterations": iterations
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================

def process_video_data(video_id: str, video_data: str, config: Dict[str, Any],
                      deletion_processor: DeletionProcessor, 
                      output_dirs: Dict[str, Path], logger: Optional[logging.Logger]) -> None:
    """Process a single video for all deletion positions."""
    positions = config['deletions']['positions']
    num_iterations = config['deletions']['num_iterations']
    number_of_segments = config['deletions']['number_of_segments']
    resume_processing = config['processing']['resume_processing']
    
    if logger:
        logger.info(f"Processing video {video_id}")
    
    # CRITICAL: Truncate the original text to first N segments before any processing
    truncated_gt = TextProcessor.limit_to_fixed_segments(video_data, number_of_segments)
    
    for position in positions:
        # Check if already processed (for resume functionality)
        if resume_processing and check_existing_output(video_id, position, output_dirs, num_iterations):
            if logger:
                logger.info(f"Skipping {video_id} - {position} (already processed)")
            continue
        
        if logger:
            logger.info(f"Generating {position} deletions for video {video_id}")
        
        # Generate iterations for this position using truncated text
        iterations = deletion_processor.generate_iterations(
            truncated_gt, position, num_iterations
        )
        
        # Save output with truncated ground truth
        save_deletion_output(video_id, truncated_gt, iterations, position, output_dirs)
        
        if logger:
            logger.info(f"Completed {position} deletions for video {video_id}")

def main():
    """Main processing function."""
    # Load configuration
    config_path = "deletion_aug_dataset.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    logger = setup_logging(config)
    
    if logger:
        logger.info("Starting deletion augmentation process")
    
    # Initialize SAT model
    if config['text_processing']['use_sat_segmentation']:
        ModelInitializer.initialize_sat()
    
    # Load data
    print("Loading ground truth data...")
    gt_data = load_ground_truth_files(config)
    print(f"Loaded {len(gt_data)} video entries")
    
    # Setup processing
    deletion_processor = DeletionProcessor(config)
    output_dirs = setup_output_directories(config)
    
    # Filter videos if specified
    selected_ids = config['processing']['selected_ids']
    if selected_ids != ["Default"]:
        gt_data = {vid: data for vid, data in gt_data.items() if vid in selected_ids}
        print(f"Processing {len(gt_data)} selected videos")
    
    # Process each video
    total_videos = len(gt_data)
    for idx, (video_id, video_data) in enumerate(gt_data.items(), 1):
        print(f"Processing video {idx}/{total_videos}: {video_id}")
        
        try:
            process_video_data(video_id, video_data, config, deletion_processor, 
                             output_dirs, logger)
        except Exception as e:
            error_msg = f"Error processing video {video_id}: {e}"
            print(error_msg)
            if logger:
                logger.error(error_msg)
            continue
    
    print("Deletion augmentation process completed!")
    if logger:
        logger.info("Deletion augmentation process completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Data augmentation script for addition experiments.
Generates augmented datasets by iteratively adding unrelated text segments.

This script processes ground truth video captions and creates multiple iterations
where unrelated text segments are progressively added at different positions:
- beginning: Added at the start of the text
- middle: Added at a random middle position  
- end: Added at the end of the text
- random: Added at completely random positions

Author: Generated from comparison scripts with addition-specific adaptations
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
    def _remove_punctuation(text: str) -> str:
        """Remove punctuation except apostrophes."""
        return text.translate(str.maketrans('', '', ''.join(PUNCTUATIONS)))
    
    @staticmethod
    def _fix_punctuation_spacing(text: str) -> str:
        """Ensure proper spacing after punctuation."""
        return re.sub(r'([.!?])(?=[^\s])', r'\1 ', text)
    
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
    log_file = logs_dir / f"addition_aug_{timestamp}.log"
    
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

def load_unrelated_segments(config: Dict[str, Any]) -> List[str]:
    """Load unrelated text segments for addition."""
    segments_file = config['data']['unrelated_segments_file']
    
    try:
        with open(segments_file, 'r', encoding='utf-8') as f:
            segments = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(segments)} unrelated segments")
        return segments
        
    except FileNotFoundError:
        print(f"Warning: Unrelated segments file not found: {segments_file}")
        return []

# ============================================================================
# ADDITION TRANSFORMATIONS
# ============================================================================

class AdditionProcessor:
    """Handles iterative addition of unrelated segments."""
    
    def __init__(self, unrelated_segments: List[str], config: Dict[str, Any]):
        self.unrelated_segments = unrelated_segments
        self.config = config
        self.random_seed = config['additions']['random_seed']
        self.add_segments = config['additions']['add_segments']
        random.seed(self.random_seed)
    
    def add_unrelated_segments_at_beginning(self, text: str, add_count: int = None) -> str:
        """Insert unrelated segments at the beginning."""
        if add_count is None:
            add_count = self.add_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        chosen = random.sample(self.unrelated_segments, 
                             k=min(add_count, len(self.unrelated_segments)))
        final_segments = chosen + current_segments
        return TextProcessor.join_segments(final_segments)
    
    def add_unrelated_segments_in_middle(self, text: str, add_count: int = None) -> str:
        """Insert unrelated segments in the middle."""
        if add_count is None:
            add_count = self.add_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        chosen = random.sample(self.unrelated_segments, 
                             k=min(add_count, len(self.unrelated_segments)))
        mid_index = len(current_segments) // 2
        final_segments = current_segments[:mid_index] + chosen + current_segments[mid_index:]
        return TextProcessor.join_segments(final_segments)
    
    def add_unrelated_segments_at_end(self, text: str, add_count: int = None) -> str:
        """Append unrelated segments at the end."""
        if add_count is None:
            add_count = self.add_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        chosen = random.sample(self.unrelated_segments, 
                             k=min(add_count, len(self.unrelated_segments)))
        final_segments = current_segments + chosen
        return TextProcessor.join_segments(final_segments)
    
    def add_unrelated_segments_randomly(self, text: str, add_count: int = None) -> str:
        """Insert unrelated segments at random positions."""
        if add_count is None:
            add_count = self.add_segments
            
        current_segments = TextProcessor.sat_segmenter(text)
        if not current_segments:
            return text
        
        chosen_segments = random.sample(self.unrelated_segments, 
                                      k=min(add_count, len(self.unrelated_segments)))
        final_segments = list(current_segments)
        
        # Pick random insertion points
        insertion_points = sorted(
            random.sample(range(len(final_segments) + 1), 
                         k=min(add_count, len(final_segments) + 1)), 
            reverse=False
        )
        
        inserted_count = 0
        for i, insert_pt in enumerate(insertion_points):
            final_segments.insert(insert_pt + inserted_count, chosen_segments[i])
            inserted_count += 1
        
        return TextProcessor.join_segments(final_segments)
    
    def generate_iterations(self, truncated_text: str, position: str, 
                          num_iterations: int) -> Dict[str, str]:
        """
        Generate multiple iterations with cumulative additions.
        
        Args:
            truncated_text: Pre-truncated ground truth text
            position: Position type for additions
            num_iterations: Number of iterations to generate
            
        Returns:
            Dictionary mapping iteration numbers to augmented texts
        """
        iterations = {}
        current_text = truncated_text
        
        for i in range(1, num_iterations + 1):
            # Add segments based on position
            if position == 'beginning':
                current_text = self.add_unrelated_segments_at_beginning(current_text)
            elif position == 'middle':
                current_text = self.add_unrelated_segments_in_middle(current_text)
            elif position == 'end':
                current_text = self.add_unrelated_segments_at_end(current_text)
            elif position == 'random':
                current_text = self.add_unrelated_segments_randomly(current_text)
            else:
                raise ValueError(f"Unknown position: {position}")
            
            iterations[str(i)] = current_text
        
        return iterations

# ============================================================================
# OUTPUT MANAGEMENT
# ============================================================================

def setup_output_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create output directory structure for addition experiment."""
    base_output_dir = Path(config['output']['output_dir'])
    positions = config['additions']['positions']
    
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

def save_addition_output(video_id: str, truncated_ground_truth: str, iterations: Dict[str, str],
                        position: str, output_dirs: Dict[str, Path]) -> None:
    """Save addition iterations to output file."""
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
                      addition_processor: AdditionProcessor, 
                      output_dirs: Dict[str, Path], logger: Optional[logging.Logger]) -> None:
    """Process a single video for all addition positions."""
    positions = config['additions']['positions']
    num_iterations = config['additions']['num_iterations']
    number_of_segments = config['additions']['number_of_segments']
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
            logger.info(f"Generating {position} additions for video {video_id}")
        
        # Generate iterations for this position using truncated text
        iterations = addition_processor.generate_iterations(
            truncated_gt, position, num_iterations
        )
        
        # Save output with truncated ground truth
        save_addition_output(video_id, truncated_gt, iterations, position, output_dirs)
        
        if logger:
            logger.info(f"Completed {position} additions for video {video_id}")

def main():
    """Main processing function."""
    # Load configuration
    config_path = "addition_aug_dataset.yaml"
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    logger = setup_logging(config)
    
    if logger:
        logger.info("Starting addition augmentation process")
    
    # Initialize SAT model
    if config['text_processing']['use_sat_segmentation']:
        ModelInitializer.initialize_sat()
    
    # Load data
    print("Loading ground truth data...")
    gt_data = load_ground_truth_files(config)
    print(f"Loaded {len(gt_data)} video entries")
    
    print("Loading unrelated segments...")
    unrelated_segments = load_unrelated_segments(config)
    
    # Setup processing
    addition_processor = AdditionProcessor(unrelated_segments, config)
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
            process_video_data(video_id, video_data, config, addition_processor, 
                             output_dirs, logger)
        except Exception as e:
            error_msg = f"Error processing video {video_id}: {e}"
            print(error_msg)
            if logger:
                logger.error(error_msg)
            continue
    
    print("Addition augmentation process completed!")
    if logger:
        logger.info("Addition augmentation process completed!")

if __name__ == "__main__":
    main()
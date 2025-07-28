#!/usr/bin/env python3
"""
Data augmentation script for comparison experiments.
Generates augmented datasets using various text transformations.

This script converts ground truth video captions into augmented versions using:
- API-based transformations (OpenAI GPT-4o)
- Local rule-based transformations
- Semantic text segmentation (SAT)

Author: Generated from notebook with refactoring
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
import openai
import yaml
import getpass
import torch
import contractions
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from wtpsplit import SaT
from typing import List
import glob

# ============================================================================
# STANDALONE SAT MODEL AND TEXT PROCESSING
# ============================================================================

# Punctuation handling - exclude apostrophes for text processing
PUNCTUATIONS = set(string.punctuation) - {"'"}

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
        
        # Preprocessing pipeline
        text = contractions.fix(text)
        text = TextProcessor._remove_punctuation(text)
        text = TextProcessor._fix_punctuation_spacing(text)
        
        # Segmentation
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

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config():
    """Load configuration from YAML file with fallback to defaults."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "comparison_aug_dataset.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    else:
        print("Configuration file not found, using defaults")
        return get_default_config()

def get_default_config():
    """Return default configuration."""
    return {
        'data': {
            'gt_file_path': "../raw_dataset/*.json",
            'prompts_file_path': "./prompts-chaining.json",
            'unrelated_segments_file': "./unrelated_segments.txt"
        },
        'output': {
            'output_dir': "../aug_dataset",
            'logs_dir': "./logs"
        },
        'api': {
            'model': "gpt-4o",
            'max_tokens': 10000,
            'temperature': 0.9,
            'max_retries': 3,
            'retry_sleep': 5.0
        },
        'processing': {
            'log_details': True,
            'enable_logging': True,
            'resume_processing': True,
            'selected_ids': ["Default"],
            'test_case_ids': ["Default"]
        },
        'transformations': {
            'random_seed': 42,
            'percentages': {
                'repeat_segments': 50.0,
                'addition_medium': 50.0,
                'addition_high': 80.0,
                'deletion_medium': 50.0,
                'deletion_high': 80.0,
                'rotation': 50.0
            }
        }
    }

def resolve_paths(config):
    """Resolve relative paths and create necessary directories."""
    script_dir = Path(__file__).parent
    
    # Resolve paths relative to script location
    config['data']['gt_file_path'] = str(script_dir / config['data']['gt_file_path'])
    config['data']['prompts_file_path'] = script_dir / config['data']['prompts_file_path']  
    config['data']['unrelated_segments_file'] = script_dir / config['data']['unrelated_segments_file']
    config['output']['output_dir'] = script_dir / config['output']['output_dir']
    config['output']['logs_dir'] = script_dir / config['output']['logs_dir']
    
    # Create output directories
    config['output']['output_dir'].mkdir(parents=True, exist_ok=True)
    config['output']['logs_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def setup_api_key(config):
    """Set up OpenAI API key from environment or interactive prompt."""
    # Priority: Environment variable -> interactive prompt
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = getpass.getpass('Enter your OpenAI API key: ')
    
    openai.api_key = api_key
    print("OpenAI API key configured")

def load_unrelated_segments(config):
    """Load unrelated segments from file."""
    segments_file = config['data']['unrelated_segments_file']
    
    if segments_file.exists():
        with open(segments_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            print(f"Loaded unrelated segments from {segments_file}")
            return content
    else:
        print(f"Warning: Unrelated segments file not found at {segments_file}")
        # Return fallback content
        return get_default_unrelated_segments()

def get_default_unrelated_segments():
    """Return default unrelated segments if file is missing."""
    return """In the neon-lit back alleys of a dystopian metropolis, a rogue hacker races against time while being chased by relentless cyborg enforcers, their footsteps echoing off rain-soaked pavement.
Under the relentless desert sun, a lone gunslinger confronts a notorious outlaw at a deserted crossroads, both men exchanging steely glances as swirling dust blurs the horizon."""

def initialize_sat():
    """Initialize SAT model for semantic text segmentation."""
    print("Initializing SAT model...")
    ModelInitializer.initialize_sat()
    print("SAT model ready")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def configure_logging(config):
    """Configure logging based on settings."""
    enable_logging = config['processing']['enable_logging']
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if enable_logging:
        logs_dir = config['output']['logs_dir']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = logs_dir / f"log_{timestamp}.log"

        logging_level = logging.INFO
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging_level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logging.root.setLevel(logging_level)
        logging.root.addHandler(file_handler)
        print(f"Logging configured to {log_filename}")
    else:
        logging_level = logging.WARNING
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()],
        )

logger = logging.getLogger(__name__)

# ============================================================================
# CATEGORY & ID HELPERS
# ============================================================================

CATEGORY_ORDER = [
    "Word-Level and Grammar",   # 1
    "Brevity and Verbosity",    # 2
    "Content Additions & Omissions",  # 3
    "Content Corruption",       # 4
    "Summarization",           # 5
    "Chronology"               # 6
]

def parse_minor_id(test_case_id: str) -> int:
    """Extract the integer portion after the first dot."""
    parts = test_case_id.split('.', 1)
    if len(parts) < 2:
        return 9999
    minor_str = parts[1].split('.', 1)[0]
    try:
        return int(minor_str)
    except ValueError:
        return 9999

def determine_category(test_case_id: str) -> str:
    """Map test_case_id to category name."""
    if test_case_id.startswith("6"):
        return "Chronology"
    elif test_case_id.startswith("5"):
        return "Summarization"
    elif test_case_id.startswith("4"):
        return "Content Corruption"
    elif test_case_id.startswith("3"):
        return "Content Additions & Omissions"
    elif test_case_id.startswith("2"):
        return "Brevity and Verbosity"
    elif test_case_id.startswith("1"):
        return "Word-Level and Grammar"
    else:
        return "Miscellaneous"

def is_api_based(test_case_id: str) -> bool:
    """Check if test_case_id requires API calls."""
    api_based_ids = {
        "1.1", "1.2",
        "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
        "3.1",
        "4.1",
        "5.1",
        "6.4"
    }
    return test_case_id in api_based_ids

def reorder_json_structure(data: dict) -> dict:
    """Re-sort categories and test cases for consistent output."""
    def category_sort_key(cat_name: str):
        if cat_name in CATEGORY_ORDER:
            return CATEGORY_ORDER.index(cat_name)
        else:
            return 9999

    data["categories"].sort(key=lambda c: category_sort_key(c["category"]))
    for cat in data["categories"]:
        cat["test_cases"].sort(key=lambda tcase: parse_minor_id(tcase["id"]))
    return data

# ============================================================================
# TEXT TRANSFORMER CLASS
# ============================================================================

class TextTransformer:
    """Handles text transformations using API and local methods."""
    
    def __init__(self, gt_text, output_filepath, prompts_filepath, config):
        self.gt_text = gt_text
        self.output_filepath = output_filepath
        self.config = config
        self.prompts_data = self.load_prompts(prompts_filepath)
        self.max_retries = config['api']['max_retries']
        self.retry_sleep = config['api']['retry_sleep']
        self.punctuations = set(string.punctuation) - set(["'"])  # keep apostrophes

    def load_prompts(self, prompts_filepath):
        """Load prompts from JSON file."""
        try:
            with open(prompts_filepath, "r", encoding='utf-8') as f:
                prompts = json.load(f)
                logger.info(f"Loaded prompts from {prompts_filepath}.")
                return prompts
        except FileNotFoundError:
            logger.error(f"Prompts file not found at {prompts_filepath}. Exiting.")
            sys.exit(1)
        except json.JSONDecodeError:
            logger.error("JSON decode error in prompts file. Exiting.")
            sys.exit(1)

    def find_prompt_by_id(self, test_case_id):
        """Find prompt data by test case ID."""
        try:
            id_float = float(test_case_id)
        except ValueError:
            logger.error(f"Invalid test_case_id format: '{test_case_id}'. Should be numeric.")
            return None

        for tcase in self.prompts_data["test_cases"]:
            if tcase["id"] == id_float:
                return tcase
        return None

    def remove_punctuation(self, text):
        """Remove punctuation except apostrophes."""
        return text.translate(str.maketrans('', '', ''.join(self.punctuations)))

    def load_json(self):
        """Load existing JSON output file."""
        try:
            with open(self.output_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.info(f"JSON file not found at {self.output_filepath}; creating a new one.")
            return self.initialize_json()
        except json.JSONDecodeError:
            logger.error("JSON decode error. Reinitializing.")
            return self.initialize_json()

    def save_json(self, data):
        """Save data to JSON output file."""
        with open(self.output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def initialize_json(self):
        """Initialize base JSON structure."""
        base_structure = {
            "ground_truth": self.gt_text,
            "categories": []
        }
        self.save_json(base_structure)
        logger.info(f"Initialized base JSON at {self.output_filepath}.")
        return base_structure

    def update_json(self, category, test_case_id, test_case_name, transformed_text):
        """Update JSON with transformation result."""
        data = self.load_json()

        # Find or create category
        found_cat = None
        for cat_obj in data["categories"]:
            if cat_obj["category"] == category:
                found_cat = cat_obj
                break
        if not found_cat:
            found_cat = {
                "category": category,
                "test_cases": []
            }
            data["categories"].append(found_cat)

        # Find or create test_case
        existing_case = None
        for tcase in found_cat["test_cases"]:
            if tcase["id"] == test_case_id:
                existing_case = tcase
                break

        if existing_case:
            existing_case["description"] = transformed_text
        else:
            found_cat["test_cases"].append({
                "id": test_case_id,
                "name": test_case_name,
                "description": transformed_text
            })

        data = reorder_json_structure(data)
        self.save_json(data)
        logger.info(f"Test case '{test_case_id}' in category '{category}' saved to JSON.")

    def process_transformation(self, transformation_func, category, test_case_id, test_case_name, **kwargs):
        """Process local transformation."""
        logger.info(f"Processing LOCAL transformation for test case {test_case_id}...")
        result = transformation_func(**kwargs)
        logger.info(f"Local transformation for {test_case_id} done.")
        self.update_json(category, test_case_id, test_case_name, result)

    def process_api_transformation(self, test_case_id, category, test_case_name):
        """Process API-based transformation."""
        logger.info(f"Processing API transformation for test case {test_case_id}...")

        prompt_data = self.find_prompt_by_id(test_case_id)
        if not prompt_data:
            logger.warning(f"No matching prompt for ID '{test_case_id}'. Skipping.")
            return None

        prompt_text = f"{prompt_data['prompt_text']}\n\nText to transform:\n{self.gt_text}"
        if self.config['processing']['log_details']:
            logger.info(f"Prompt for {test_case_id}: {prompt_text}")

        gen_text = None
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config['api']['model'],
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=self.config['api']['max_tokens'],
                    temperature=self.config['api']['temperature']
                )
                gen_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logger.error(f"OpenAI error (attempt {attempt+1} for {test_case_id}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_sleep} seconds...")
                    time.sleep(self.retry_sleep)
                else:
                    logger.error(f"Aborting test_case_id {test_case_id} after max retries.")
                    return None

        if gen_text:
            if self.config['processing']['log_details']:
                logger.info(f"Output for {test_case_id}: {gen_text}")
            logger.info(f"API transformation for {test_case_id} succeeded.")
            self.update_json(category, test_case_id, test_case_name, gen_text)
        return gen_text

    def process_api_transformation_custom_prompt(self, test_case_id, category, test_case_name, custom_prompt_text):
        """Process chained API transformation with custom prompt."""
        logger.info(f"Processing CHAINED API transformation for test case {test_case_id} using parent prompt...")

        prompt_text = f"{custom_prompt_text}\n\nText to transform:\n{self.gt_text}"
        if self.config['processing']['log_details']:
            logger.info(f"Chained prompt for {test_case_id}: {prompt_text}")

        gen_text = None
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.config['api']['model'],
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=self.config['api']['max_tokens'],
                    temperature=self.config['api']['temperature']
                )
                gen_text = response.choices[0].message.content.strip()
                break
            except Exception as e:
                logger.error(f"OpenAI error (attempt {attempt+1} for {test_case_id}): {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_sleep} seconds...")
                    time.sleep(self.retry_sleep)
                else:
                    logger.error(f"Aborting {test_case_id} after max retries.")
                    return None

        if gen_text:
            if self.config['processing']['log_details']:
                logger.info(f"Chained output for {test_case_id}: {gen_text}")
            logger.info(f"Chained API transformation for {test_case_id} succeeded.")
            self.update_json(category, test_case_id, test_case_name, gen_text)
        return gen_text

    # ========================================================================
    # LOCAL TRANSFORMATIONS
    # ========================================================================

    def complete_hallucination(self, unrelated_segments, seed=None):
        """Replace text with completely unrelated content."""
        if seed:
            random.seed(seed)
        return " ".join(unrelated_segments)

    def add_segments_to_middle(self, unrelated_segments_text, percentage=50.0, seed=None):
        """Add unrelated segments to the middle of ground truth."""
        if seed:
            random.seed(seed)
            
        # Use TextProcessor for semantic segmentation
        gt_segments = TextProcessor.sat_segmenter(self.gt_text)
        unrelated_segments = TextProcessor.sat_segmenter(unrelated_segments_text)
        
        # Calculate how many segments to add
        num_gt_segments = len(gt_segments)
        num_segments_to_add = int(num_gt_segments * (percentage / 100.0))
        
        logger.info(f"Ground truth has {num_gt_segments} semantic segments")
        logger.info(f"Adding {num_segments_to_add} segments ({percentage}%)")
        
        # Take segments from unrelated_segments
        segments_to_add = []
        while len(segments_to_add) < num_segments_to_add:
            segments_needed = num_segments_to_add - len(segments_to_add)
            if len(unrelated_segments) == 0:
                break
            segments_to_add.extend(unrelated_segments[:min(segments_needed, len(unrelated_segments))])
        
        # Insert in the middle
        middle_index = num_gt_segments // 2
        modified_segments = gt_segments[:middle_index] + segments_to_add + gt_segments[middle_index:]
        
        # Join segments with proper punctuation
        joined_text = ""
        for segment in modified_segments:
            if segment and segment[-1] in ".!?":
                joined_text += segment + " "
            else:
                joined_text += segment + ". "
        
        return joined_text.strip()

    def delete_segments_from_middle(self, percentage=50.0, seed=None):
        """Delete segments from the middle of ground truth."""
        if seed:
            random.seed(seed)
            
        # Use TextProcessor for semantic segmentation
        gt_segments = TextProcessor.sat_segmenter(self.gt_text)
        
        # Calculate how many segments to delete
        num_gt_segments = len(gt_segments)
        num_segments_to_delete = int(num_gt_segments * (percentage / 100.0))
        
        logger.info(f"Ground truth has {num_gt_segments} semantic segments")
        logger.info(f"Deleting {num_segments_to_delete} segments ({percentage}%)")
        
        # Ensure we don't delete everything
        num_segments_to_delete = min(num_segments_to_delete, num_gt_segments - 2)
        
        # Find middle point to start deletion
        middle_index = (num_gt_segments // 2) - (num_segments_to_delete // 2)
        if middle_index < 0:
            middle_index = 0
        
        # Delete segments from the middle
        modified_segments = gt_segments[:middle_index] + gt_segments[middle_index + num_segments_to_delete:]
        
        # Join segments with proper punctuation
        joined_text = ""
        for segment in modified_segments:
            if segment and segment[-1] in ".!?":
                joined_text += segment + " "
            else:
                joined_text += segment + ". "
        
        return joined_text.strip()

    def transform_paragraph(self, paragraph: str, op_code: str) -> str:
        """Transform paragraph order (reverse or jumble)."""
        segs = TextProcessor.sat_segmenter(paragraph)
        if not segs:
            return ""
        if op_code == "reverse":
            return " ".join(segs[::-1])
        elif op_code == "jumble":
            random.shuffle(segs)
            return " ".join(segs)
        else:
            raise ValueError("op_code must be 'reverse' or 'jumble'")

    def rotate_paragraph(self, paragraph: str, percentage: float, position: str="start") -> str:
        """Rotate paragraph segments."""
        segs = TextProcessor.sat_segmenter(paragraph)
        if not segs or percentage <= 0:
            return paragraph
        chunk_size = math.ceil(len(segs) * (percentage / 100.0))
        chunk_size = min(chunk_size, len(segs))
        if position == "start":
            chunk = segs[:chunk_size]
            remainder = segs[chunk_size:]
            rotated = remainder + chunk
        elif position == "end":
            chunk = segs[-chunk_size:]
            remainder = segs[:-chunk_size]
            rotated = chunk + remainder
        else:
            raise ValueError("position must be 'start' or 'end'")
        return " ".join(rotated)

    def neighbor_swap(self, paragraph: str, op_code: str) -> str:
        """Swap neighboring segments in different patterns."""
        segs = TextProcessor.sat_segmenter(paragraph)
        if not segs:
            return ""
        
        if op_code == "adjacent":
            # Swap pairs: [1,2,3,4,5] → [2,1,4,3,5]
            for i in range(0, len(segs) - 1, 2):
                segs[i], segs[i + 1] = segs[i + 1], segs[i]
        
        elif op_code == "triplet":
            # Swap first and last in triplets: [1,2,3,4,5,6] → [3,2,1,6,5,4]
            for i in range(0, len(segs), 3):
                if i + 2 < len(segs):  # Full triplet
                    segs[i], segs[i + 2] = segs[i + 2], segs[i]
                elif i + 1 < len(segs):  # Pair remaining
                    segs[i], segs[i + 1] = segs[i + 1], segs[i]
        
        else:
            raise ValueError("op_code must be 'adjacent' or 'triplet'")
        
        return " ".join(segs)

# ============================================================================
# LOCAL TRANSFORMATIONS DISPATCHER
# ============================================================================

def perform_local_transformation(transformer, test_case_id, category, test_case_name, unrelated_segments):
    """Dispatch local transformations based on test case ID."""
    config = transformer.config
    
    local_transformations = {
        "3.2": {
            "func": lambda: transformer.add_segments_to_middle(
                unrelated_segments_text=unrelated_segments,
                percentage=config['transformations']['percentages']['addition_medium'],
                seed=config['transformations']['random_seed']
            ),
            "name": "Addition (~50%)"
        },
        "3.3": {
            "func": lambda: transformer.add_segments_to_middle(
                unrelated_segments_text=unrelated_segments,
                percentage=config['transformations']['percentages']['addition_high'],
                seed=config['transformations']['random_seed']
            ),
            "name": "Addition (~80%)"
        },
        "3.4": {
            "func": lambda: transformer.delete_segments_from_middle(
                percentage=config['transformations']['percentages']['deletion_medium'],
                seed=config['transformations']['random_seed']
            ),
            "name": "Deletion (~50%)"
        },
        "3.5": {
            "func": lambda: transformer.delete_segments_from_middle(
                percentage=config['transformations']['percentages']['deletion_high'],
                seed=config['transformations']['random_seed']
            ),
            "name": "Deletion (~80%)"
        },
        "4.2": {
            "func": lambda: transformer.complete_hallucination(
                unrelated_segments=[
                    "The film starts with a man waking up in a small apartment. He prepares coffee and checks his phone for messages. There are none, so he heads out to catch a bus. On the way, he passes a neighbor who greets him briefly. At the bus stop, he notices a group of students talking about an upcoming test. He boards the bus and sits near the back, watching the city pass by outside the window. After a few stops, he gets off near a large office building. He enters the lobby and goes through security. He rides an elevator to the tenth floor, where he works as an assistant. His manager asks him to prepare some documents for a meeting later in the day. He collects files from different departments and organizes them in a conference room. Another employee asks him for help with a software issue, so he takes a few minutes to fix it. Then he returns to the conference room to double-check everything before the meeting begins. The meeting starts at noon. Several people join via video call, and others sit around the table. The manager outlines the project goals, and each team member shares updates on their tasks. There are questions about deadlines and budgets, but no big surprises. After an hour, they end the call. The manager thanks everyone and leaves. The assistant cleans up the room, collects leftover notes, and heads back to his desk. During lunch, he walks to a nearby café, where he orders a simple meal. He eats alone and checks social media on his phone. After lunch, he returns to the office and finishes smaller tasks, such as sending emails and filing paperwork. He also helps a coworker carry boxes of supplies to another floor. Later in the afternoon, the manager calls him in to discuss next week's schedule. They go over a few changes, and the assistant updates the calendar. When the workday ends, he logs off his computer and leaves the building. Outside, he walks a few blocks to meet a friend. They chat about weekend plans and decide to see a film on Saturday. The assistant then heads home on a crowded bus. At his apartment, he sorts his mail, feeds a pet cat, and warms up leftovers for dinner. He watches a short news segment on TV, then checks his phone again. A message from his friend confirms their plan for the weekend. He feels relieved to have something to look forward to. The film ends with him preparing for bed, setting an alarm, and turning off the lights. He reflects briefly on the day, thinking about his tasks, his quiet home, and the plans ahead.",
                ],
                seed=None
            ),
            "name": "Complete Corruption"
        },
        "6.1": {
            "func": lambda: transformer.transform_paragraph(
                paragraph=transformer.gt_text,
                op_code="reverse"
            ),
            "name": "Reverse Segments Order"
        },
        "6.2": {
            "func": lambda: transformer.neighbor_swap(
                paragraph=transformer.gt_text,
                op_code="adjacent"
            ),
            "name": "Adjacent Neighbor Swap"
        },
        "6.3": {
            "func": lambda: transformer.rotate_paragraph(
                paragraph=transformer.gt_text,
                percentage=config['transformations']['percentages']['rotation'],
                position="start"
            ),
            "name": "Rotate Half Paragraph"
        }
    }

    mapping = local_transformations.get(test_case_id)
    if not mapping:
        logger.error(f"No local transformation defined for {test_case_id}; skipping.")
        return

    func = mapping["func"]
    params = mapping.get("params", {})
    transformer.process_transformation(
        transformation_func=func,
        category=category,
        test_case_id=test_case_id,
        test_case_name=mapping["name"],
        **params
    )

# ============================================================================
# MAIN EXECUTION LOGIC
# ============================================================================

def get_processing_status(config, ground_truths, test_case_ids):
    """Check which files and test cases are already processed."""
    output_dir = Path(config['output']['output_dir'])
    processed_files = {}
    
    if not output_dir.exists():
        return {}
    
    for video_id in ground_truths.keys():
        output_file = output_dir / f"{video_id}.json"
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                # Check which test cases are complete by examining the categories structure
                completed_test_cases = set()
                if 'categories' in existing_data:
                    for category in existing_data['categories']:
                        if 'test_cases' in category:
                            for test_case in category['test_cases']:
                                test_id = test_case.get('id')
                                description = test_case.get('description', '')
                                # Test case is complete if it has an ID and non-empty description
                                if test_id and description and description.strip():
                                    completed_test_cases.add(test_id)
                
                processed_files[video_id] = {
                    'file_exists': True,
                    'completed_test_cases': completed_test_cases,
                    'remaining_test_cases': set(test_case_ids) - completed_test_cases
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading {video_id}.json: {e}")
                processed_files[video_id] = {
                    'file_exists': True,
                    'completed_test_cases': set(),
                    'remaining_test_cases': set(test_case_ids)
                }
        else:
            processed_files[video_id] = {
                'file_exists': False,
                'completed_test_cases': set(),
                'remaining_test_cases': set(test_case_ids)
            }
    
    return processed_files

def run_transformations(config, unrelated_segments):
    """Execute transformations in category-based order."""
    
    # Load ground truths from all matching JSON files
    gt_file_pattern = str(config['data']['gt_file_path'])
    gt_files = glob.glob(gt_file_pattern)
    
    if not gt_files:
        logger.error(f"No ground truth files found matching pattern: {gt_file_pattern}. Exiting.")
        return
    
    logger.info(f"Found {len(gt_files)} ground truth files: {[Path(f).name for f in gt_files]}")
    
    # Combine all ground truth files into one dictionary
    ground_truths = {}
    for gt_file in sorted(gt_files):  # Sort for consistent processing order
        try:
            with open(gt_file, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                logger.info(f"Loaded {len(file_data)} entries from {Path(gt_file).name}")
                ground_truths.update(file_data)
        except FileNotFoundError:
            logger.error(f"Ground truth file not found: {gt_file}")
            continue
        except json.JSONDecodeError:
            logger.error(f"JSON decode error in file: {gt_file}")
            continue
    
    if not ground_truths:
        logger.error("No valid ground truth data loaded. Exiting.")
        return
    
    logger.info(f"Total ground truth entries loaded: {len(ground_truths)}")

    # Process selected IDs
    selected_ids = config['processing']['selected_ids']
    if selected_ids == ["Default"]:
        selected_ids = list(ground_truths.keys())
        logger.info("selected_ids set to all available IDs.")
    else:
        invalid = [s for s in selected_ids if s not in ground_truths]
        if invalid:
            logger.error(f"Invalid selected_ids: {invalid}")
            selected_ids = [s for s in selected_ids if s in ground_truths]
            if not selected_ids:
                logger.error("No valid selected_ids. Exiting.")
                return
        logger.info(f"Processing selected_ids: {selected_ids}")

    # Load prompts
    try:
        with open(config['data']['prompts_file_path'], 'r', encoding='utf-8') as f:
            prompts_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {config['data']['prompts_file_path']}. Exiting.")
        return
    except json.JSONDecodeError:
        logger.error("JSON decode error in prompts file.")
        return

    # Define test case types
    all_local_test_case_ids = {"3.2", "3.3", "3.4", "3.5", "4.2", "6.1", "6.2", "6.3"}
    
    chain_prompt_map = {
        "2.4": "2.3"
    }

    rename_map = {}

    # Process test case IDs
    test_case_ids = config['processing']['test_case_ids']
    if test_case_ids == ["Default"]:
        test_case_ids_api = [str(tc["id"]) for tc in prompts_data["test_cases"]]
        test_case_ids_local = list(all_local_test_case_ids)
        all_test_case_ids = test_case_ids_api + test_case_ids_local
        logger.info("test_case_ids set to all (API + local).")
    else:
        all_test_case_ids = test_case_ids
        test_case_ids_api = [tc for tc in test_case_ids if is_api_based(tc)]
        test_case_ids_local = [tc for tc in test_case_ids if tc in all_local_test_case_ids]
        
        # Validate API test case IDs
        valid_api_ids = {str(tc["id"]) for tc in prompts_data["test_cases"]}
        invalid_api = [tc for tc in test_case_ids_api if tc not in valid_api_ids]
        if invalid_api:
            logger.error(f"Some requested API-based test IDs not in prompts.json: {invalid_api}")
            test_case_ids_api = [tc for tc in test_case_ids_api if tc in valid_api_ids]
        
        # Validate local test case IDs
        invalid_local = [tc for tc in test_case_ids_local if tc not in all_local_test_case_ids]
        if invalid_local:
            logger.error(f"Some requested local test IDs not valid: {invalid_local}")
            test_case_ids_local = [tc for tc in test_case_ids_local if tc in all_local_test_case_ids]
        
        if not test_case_ids_api and not test_case_ids_local:
            logger.error("No valid test case IDs found after filtering. Exiting.")
            return
        
        if test_case_ids_api:
            logger.info(f"API test_case_ids: {test_case_ids_api}")
        if test_case_ids_local:
            logger.info(f"Local test_case_ids: {test_case_ids_local}")
    
    # Handle resume processing
    resume_processing = config['processing'].get('resume_processing', True)
    output_dir = Path(config['output']['output_dir'])
    
    if resume_processing:
        logger.info("Resume mode enabled. Checking existing processed files...")
        processing_status = get_processing_status(config, ground_truths, all_test_case_ids)
        
        # Filter out fully completed files and get remaining test cases
        remaining_ids = []
        files_to_process = {}
        
        for video_id in selected_ids:
            status = processing_status.get(video_id, {'remaining_test_cases': set(all_test_case_ids)})
            if status['remaining_test_cases']:
                remaining_ids.append(video_id)
                files_to_process[video_id] = status['remaining_test_cases']
                if status['completed_test_cases']:
                    completed = sorted(status['completed_test_cases'])
                    remaining = sorted(status['remaining_test_cases'])
                    logger.info(f"{video_id}.json: Completed {completed}, remaining {remaining}")
                else:
                    logger.info(f"{video_id}.json: No previous progress found")
            else:
                logger.info(f"{video_id}.json: Already fully processed, skipping")
        
        selected_ids = remaining_ids
        if len(remaining_ids) < len(selected_ids):
            skipped = len(selected_ids) - len(remaining_ids)
            logger.info(f"Resume mode: {skipped} files fully completed, {len(remaining_ids)} files remaining")
        else:
            logger.info(f"Resume mode: {len(remaining_ids)} files to process")
    else:
        logger.info("Fresh start mode. Will overwrite existing files.")
        files_to_process = {vid: set(all_test_case_ids) for vid in selected_ids}
        # Clear output directory if exists
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
            logger.info("Cleared existing output directory")
    
    if not selected_ids:
        logger.info("All files already processed. Nothing to do.")
        return

    # Organize by category
    category_to_testcases = defaultdict(list)
    combined_ids = sorted(
        set(test_case_ids_api + test_case_ids_local),
        key=lambda x: (determine_category(x), parse_minor_id(x))
    )
    for tcid in combined_ids:
        cat = determine_category(tcid)
        category_to_testcases[cat].append(tcid)

    for cat in category_to_testcases:
        category_to_testcases[cat].sort(key=parse_minor_id)

    CATEGORY_ORDER_LIST = [
        "Word-Level and Grammar",
        "Brevity and Verbosity",
        "Content Additions & Omissions",
        "Content Corruption",
        "Summarization",
        "Chronology"
    ]

    chain_outputs = {}

    # Process each selected ID
    for i, sid in enumerate(selected_ids, 1):
        if sid not in ground_truths:
            logger.error(f"No ground truth text for ID {sid}")
            continue
        
        logger.info(f"Processing file {i}/{len(selected_ids)}: {sid}.json")
        
        # Show remaining test cases for this file if resuming
        if resume_processing and sid in files_to_process:
            remaining = sorted(files_to_process[sid])
            if len(remaining) < len(all_test_case_ids):
                logger.info(f"  Resuming with test cases: {remaining}")
        
        original_gt_text = ground_truths[sid]
        out_file = config['output']['output_dir'] / f"{sid}.json"

        transformer = TextTransformer(
            gt_text=original_gt_text,
            output_filepath=out_file,
            prompts_filepath=config['data']['prompts_file_path'],
            config=config
        )

        for cat in CATEGORY_ORDER_LIST:
            tcase_ids = category_to_testcases.get(cat, [])
            if not tcase_ids:
                continue

            for tcid in tcase_ids:
                if tcid in rename_map:
                    tcase_name = rename_map[tcid]
                else:
                    found_prompt = next((x for x in prompts_data["test_cases"] if str(x["id"]) == tcid), None)
                    tcase_name = found_prompt["name"] if found_prompt else tcid

                # Skip if already processed (resume mode)
                if resume_processing and tcid not in files_to_process.get(sid, set()):
                    logger.debug(f"  Skipping test case {tcid} (already completed)")
                    continue
                
                logger.info(f"  Processing test case: {tcid} ({tcase_name})")
                
                if tcid in test_case_ids_local:
                    perform_local_transformation(transformer, tcid, cat, tcase_name, unrelated_segments)
                    continue

                if tcid in chain_prompt_map:
                    parent_id = chain_prompt_map[tcid]
                    data = transformer.load_json()
                    parent_output = None
                    for cat_obj in data.get("categories", []):
                        for tcase in cat_obj.get("test_cases", []):
                            if tcase["id"] == parent_id:
                                parent_output = tcase.get("description")
                                break
                        if parent_output:
                            break
                    if not parent_output:
                        logger.info(f"Parent output for '{parent_id}' not found in JSON. Sending API call to generate it.")
                        parent_output = transformer.process_api_transformation(parent_id, cat, rename_map.get(parent_id, parent_id))
                        if parent_output:
                            chain_outputs[parent_id] = parent_output
                    if not parent_output:
                        logger.warning(f"No chained output found for parent '{parent_id}' - skipping chained test case '{tcid}'.")
                        continue
                    # Save the original ground truth
                    original_gt_text = transformer.gt_text
                    # Overwrite with parent's output for this chained transformation
                    transformer.gt_text = parent_output
                    parent_prompt_data = transformer.find_prompt_by_id(parent_id)
                    if not parent_prompt_data:
                        logger.warning(f"No parent prompt found for '{parent_id}'. Skipping '{tcid}'.")
                        # Restore the original ground truth
                        transformer.gt_text = original_gt_text
                        continue
                    new_result = transformer.process_api_transformation_custom_prompt(
                        test_case_id=tcid,
                        category=cat,
                        test_case_name=tcase_name,
                        custom_prompt_text=parent_prompt_data["prompt_text"]
                    )
                    # Restore the original ground truth after chaining
                    transformer.gt_text = original_gt_text
                    if new_result:
                        chain_outputs[tcid] = new_result
                else:
                    new_result = transformer.process_api_transformation(tcid, cat, tcase_name)
                    if new_result:
                        chain_outputs[tcid] = new_result

    logger.info("All transformations complete.")

def main():
    """Main execution function."""
    print("=" * 80)
    print("Data Augmentation Script for Comparison Experiments")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    config = resolve_paths(config)
    
    # Setup components
    setup_api_key(config)
    configure_logging(config)
    
    # Initialize SAT model
    initialize_sat()
    
    # Load unrelated segments
    unrelated_segments = load_unrelated_segments(config)
    
    # Run transformations
    print(f"Starting transformations...")
    print(f"Selected IDs: {config['processing']['selected_ids']}")
    print(f"Test case IDs: {config['processing']['test_case_ids']}")
    print("-" * 80)
    
    run_transformations(config, unrelated_segments)
    
    print("-" * 80)
    print("Processing complete!")

if __name__ == "__main__":
    main()
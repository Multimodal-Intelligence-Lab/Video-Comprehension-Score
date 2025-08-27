#!/usr/bin/env python3
"""
Claude AI processing script for transforming video descriptions.
Processes CSV files containing video descriptions and transforms them using Claude AI.

This script reads CSV files with "Description" columns, constructs prompts,
and uses Claude AI to transform the content into coherent movie descriptions.

Author: Converted from Jupyter notebook
"""

import os
import glob
import csv
import json
import sys
import time
import logging
import yaml
import getpass
from datetime import datetime
from pathlib import Path
from anthropic import Anthropic

# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================

def load_config():
    """Load configuration from YAML file with fallback to defaults."""
    script_dir = Path(__file__).parent
    config_path = script_dir / "base.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
    else:
        print("Configuration file not found, exiting")
        sys.exit(1)

def resolve_paths(config):
    """Resolve relative paths and create necessary directories."""
    script_dir = Path(__file__).parent
    
    # Resolve paths relative to script location
    config['data']['input_base_folder'] = script_dir / config['data']['input_base_folder']
    config['data']['output_dir'] = script_dir / config['data']['output_dir']
    config['output']['logs_dir'] = script_dir / config['output']['logs_dir']
    
    # Create output directories
    config['data']['output_dir'].mkdir(parents=True, exist_ok=True)
    config['output']['logs_dir'].mkdir(parents=True, exist_ok=True)
    
    return config

def get_api_key(config):
    """Get API key from config or prompt user if empty."""
    api_key = config.get('claude_api_key', '')
    
    if not api_key or api_key.strip() == '':
        api_key = getpass.getpass('Enter your Claude API key: ')
        if not api_key.strip():
            print("API key is required. Exiting.")
            sys.exit(1)
    
    return api_key

def discover_input_folders(config):
    """Auto-discover all rawGT3_part* folders in the input base directory."""
    base_folder = config['data']['input_base_folder']
    
    if not base_folder.exists():
        print(f"Input base folder does not exist: {base_folder}")
        sys.exit(1)
    
    # Find all rawGT3_part* folders
    part_folders = []
    for item in base_folder.iterdir():
        if item.is_dir() and item.name.startswith('rawGT3_part'):
            part_folders.append(item)
    
    part_folders.sort()  # Sort to process in order
    
    if not part_folders:
        print(f"No rawGT3_part* folders found in {base_folder}")
        sys.exit(1)
    
    print(f"Found {len(part_folders)} folders to process:")
    for folder in part_folders:
        print(f"  - {folder.name}")
    
    return part_folders

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
        log_filename = logs_dir / f"claude_{timestamp}.log"

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
# CLAUDE AI CLIENT
# ============================================================================

class ClaudeClient:
    """Handles Claude AI API interactions."""
    
    def __init__(self, config, api_key):
        self.config = config
        self.api_key = api_key
        self.model = config['claude_model']
        self.max_retries = config['api']['max_retries']
        self.retry_sleep = config['api']['retry_sleep']
        
        # Initialize the Claude client
        self.client = Anthropic(api_key=self.api_key)
        
        logger.info(f"Claude client initialized with model: {self.model}")

    def get_response(self, prompt):
        """
        Call Claude AI's messages endpoint to get a response.
        """
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.config['api']['max_tokens'],
            temperature=self.config['api']['temperature'],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text.strip()

# ============================================================================
# CSV PROCESSING
# ============================================================================

def process_csv_files(config, claude_client, folder_path, output_json_file):
    """
    Process all CSV files in a folder. For each CSV:
      1) Read the "Description" column in order.
      2) Construct a single prompt.
      3) Call Claude AI with retries on error.
      4) Save partial progress in a JSON file so the script can be resumed.
      
      The JSON key for each file is extracted from the filename
      after the last underscore and before the ".csv" extension.
    """
    
    # 1) Load existing results (if any)
    if config['processing']['resume_processing'] and os.path.exists(output_json_file):
        with open(output_json_file, "r", encoding="utf-8") as jf:
            try:
                results = json.load(jf)
            except json.JSONDecodeError:
                results = {}
        logger.info(f"Loaded existing results from '{output_json_file}'.")
    else:
        results = {}
        logger.info(f"No existing '{output_json_file}' found or resume disabled. Starting fresh.")
    
    # 2) Get all CSV files
    csv_files = sorted(glob.glob(os.path.join(str(folder_path), "*.csv")))
    logger.info(f"Found {len(csv_files)} CSV file(s) in folder '{folder_path}'.")

    if not csv_files:
        logger.warning(f"No CSV files found in {folder_path}")
        return

    # 3) Process each CSV file
    for csv_file in csv_files:
        # Extract the file ID from filename
        filename = os.path.basename(csv_file)
        filename_no_ext = os.path.splitext(filename)[0]
        file_id = filename_no_ext.split('_')[-1]

        # Skip if already processed
        if file_id in results:
            logger.info(f"Skipping file with ID '{file_id}' ('{csv_file}') - already in JSON results.")
            continue
        
        logger.info(f"Processing file with ID '{file_id}': '{csv_file}'")

        # Read the CSV and extract the "Description" column
        descriptions = []
        try:
            with open(csv_file, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter=",")
                for row_number, row in enumerate(reader, start=1):
                    if "Description" in row:
                        descriptions.append(row["Description"])
                    else:
                        logger.warning(f"Warning: 'Description' column not found in row {row_number} of {csv_file}")
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_file}: {e}")
            continue

        if not descriptions:
            logger.warning(f"No descriptions found in {csv_file}")
            continue

        # Build the prompt
        prompt = config['prompt']['text']
        for i, desc in enumerate(descriptions, start=1):
            prompt += f"{i}\t{desc}\n"

        # 4) Call Claude AI with retries
        attempt = 0
        response_text = None
        while attempt < claude_client.max_retries:
            try:
                response_text = claude_client.get_response(prompt)
                break
            except Exception as e:
                attempt += 1
                logger.error(f"Error on attempt {attempt} for file ID '{file_id}': {e}")
                if attempt < claude_client.max_retries:
                    logger.info(f"Retrying (attempt {attempt+1}/{claude_client.max_retries})...")
                    time.sleep(claude_client.retry_sleep)
                else:
                    results[file_id] = f"Error after {claude_client.max_retries} attempts: {e}"
                    with open(output_json_file, "w", encoding='utf-8') as json_file:
                        json.dump(results, json_file, indent=4, ensure_ascii=False)
                    logger.error(f"Failed to process file ID '{file_id}' after {claude_client.max_retries} attempts")
                    continue

        # 5) Store result
        if response_text:
            results[file_id] = response_text
            logger.info(f"[File ID '{file_id}'] Stored Claude AI response in results dictionary.")
            
            if config['processing']['log_details']:
                logger.info(f"Response for {file_id}: {response_text[:200]}...")

            # 6) Save partial progress
            with open(output_json_file, "w", encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)
            logger.info(f"Progress saved after processing file ID '{file_id}'.")

    logger.info(f"All CSV files processed for folder '{folder_path.name}'. Results saved to '{output_json_file}'.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("Claude AI Video Description Processing Script")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    config = resolve_paths(config)
    
    # Setup logging
    configure_logging(config)
    
    # Get API key
    api_key = get_api_key(config)
    
    # Discover input folders
    input_folders = discover_input_folders(config)
    
    # Initialize Claude client
    claude_client = ClaudeClient(config, api_key)
    
    logger.info("Starting CSV processing and Claude AI querying...")
    logger.info(f"Found {len(input_folders)} folders to process")
    
    # Generate output filename - simple name for all parts combined
    output_filename = "claude.json"
    output_json_file = config['data']['output_dir'] / output_filename
    
    # Process each folder
    for i, folder_path in enumerate(input_folders, 1):
        print(f"\nProcessing folder {i}/{len(input_folders)}: {folder_path.name}")
        logger.info(f"Processing folder {i}/{len(input_folders)}: {folder_path.name}")
        
        logger.info(f"Output file: {output_json_file}")
        
        try:
            process_csv_files(config, claude_client, folder_path, str(output_json_file))
            print(f"Completed processing {folder_path.name}")
        except Exception as e:
            logger.error(f"Failed to process folder {folder_path.name}: {e}")
            print(f"Failed to process folder {folder_path.name}: {e}")
            continue
    
    logger.info("All folders processed.")
    print("=" * 80)
    print("Processing complete!")

if __name__ == "__main__":
    main()
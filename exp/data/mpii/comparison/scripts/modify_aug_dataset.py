#!/usr/bin/env python3
"""
Script to modify augmented dataset JSON files.
Removes specified test cases and renumbers others according to requirements.
"""

import os
import json
import glob
from pathlib import Path

def modify_json_file(file_path):
    """Modify a single JSON file according to the requirements."""
    print(f"Processing: {file_path}")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Test cases to remove
    test_cases_to_remove = {"3.2", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6"}
    
    # Renumbering mappings
    category_3_mapping = {
        "3.3": "3.2",
        "3.4": "3.3", 
        "3.5": "3.4",
        "3.6": "3.5"
    }
    
    category_4_mapping = {
        "4.7": "4.1",
        "4.8": "4.2"
    }
    
    # Process each category
    if 'categories' in data:
        for category in data['categories']:
            if 'test_cases' in category:
                # Filter out test cases to remove and update remaining ones
                updated_test_cases = []
                
                for test_case in category['test_cases']:
                    test_id = test_case.get('id')
                    
                    # Skip test cases that should be removed
                    if test_id in test_cases_to_remove:
                        print(f"  Removing test case: {test_id}")
                        continue
                    
                    # Handle category 3 renumbering
                    if test_id in category_3_mapping:
                        old_id = test_id
                        test_case['id'] = category_3_mapping[test_id]
                        print(f"  Renumbered: {old_id} → {test_case['id']}")
                    
                    # Handle category 4 renumbering  
                    elif test_id in category_4_mapping:
                        old_id = test_id
                        test_case['id'] = category_4_mapping[test_id]
                        print(f"  Renumbered: {old_id} → {test_case['id']}")
                    
                    # Handle test case 6.2 special update
                    elif test_id == "6.2":
                        test_case['name'] = "Adjacent Neighbor Swap"
                        test_case['description'] = " "
                        print(f"  Updated test case 6.2: name and description")
                    
                    updated_test_cases.append(test_case)
                
                # Update the test cases list
                category['test_cases'] = updated_test_cases
    
    # Save modified JSON
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Completed: {file_path}")

def main():
    """Main function to process all JSON files in aug_dataset directory."""
    print("=" * 60)
    print("Modifying Augmented Dataset JSON Files")
    print("=" * 60)
    
    # Get the aug_dataset directory path
    script_dir = Path(__file__).parent
    aug_dataset_dir = script_dir.parent / "aug_dataset"
    
    if not aug_dataset_dir.exists():
        print(f"Error: aug_dataset directory not found at {aug_dataset_dir}")
        return
    
    # Find all JSON files in the directory
    json_pattern = str(aug_dataset_dir / "*.json")
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"No JSON files found in {aug_dataset_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    print("-" * 60)
    
    # Process each JSON file
    for json_file in sorted(json_files):
        try:
            modify_json_file(json_file)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    print("-" * 60)
    print("Processing complete!")
    
    print("\nSummary of changes:")
    print("- Removed test cases: 3.2, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6")
    print("- Renumbered category 3: 3.3→3.2, 3.4→3.3, 3.5→3.4, 3.6→3.5") 
    print("- Renumbered category 4: 4.7→4.1, 4.8→4.2")
    print("- Updated test case 6.2: name='Adjacent Neighbor Swap', description=' '")

if __name__ == "__main__":
    main()
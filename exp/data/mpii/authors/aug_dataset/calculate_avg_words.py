#!/usr/bin/env python3
import json
import sys

def count_words(text):
    """Count words in a text string, handling basic word separation."""
    return len(text.split())

def main():
    try:
        # Load the JSON file
        with open('chatgpt.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded JSON file with {len(data)} entries")
        
        # Count words in each description
        word_counts = []
        for key, description in data.items():
            word_count = count_words(description)
            word_counts.append(word_count)
        
        # Calculate statistics
        total_descriptions = len(word_counts)
        total_words = sum(word_counts)
        average_words = total_words / total_descriptions if total_descriptions > 0 else 0
        
        # Find min and max for additional context
        min_words = min(word_counts) if word_counts else 0
        max_words = max(word_counts) if word_counts else 0
        
        # Display results
        print("\n" + "="*50)
        print("WORD COUNT ANALYSIS RESULTS")
        print("="*50)
        print(f"Total descriptions: {total_descriptions}")
        print(f"Total words across all descriptions: {total_words:,}")
        print(f"Average words per description: {average_words:.2f}")
        print(f"Minimum words in a description: {min_words}")
        print(f"Maximum words in a description: {max_words}")
        print("="*50)
        
        # Show some examples for verification
        print("\nSample word counts from first few entries:")
        count = 0
        for key, description in data.items():
            if count >= 5:  # Show first 5 examples
                break
            words = count_words(description)
            preview = description[:100] + "..." if len(description) > 100 else description
            print(f"Entry {key}: {words} words - '{preview}'")
            count += 1
            
    except FileNotFoundError:
        print("Error: chatgpt.json file not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
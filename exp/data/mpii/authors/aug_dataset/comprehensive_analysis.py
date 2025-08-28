#!/usr/bin/env python3
import json
import re
import string
from collections import Counter, defaultdict
import statistics
import sys

def count_sentences(text):
    """Count sentences using basic punctuation."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])

def count_characters(text):
    """Count characters (total and without spaces)."""
    return len(text), len(text.replace(' ', ''))

def analyze_vocabulary(text):
    """Analyze vocabulary complexity."""
    words = text.lower().split()
    # Remove punctuation
    words = [''.join(c for c in word if c not in string.punctuation) for word in words]
    words = [word for word in words if word]  # Remove empty strings
    
    unique_words = set(words)
    return {
        'total_words': len(words),
        'unique_words': len(unique_words),
        'vocabulary_richness': len(unique_words) / len(words) if words else 0
    }

def get_common_words(all_texts, top_n=20):
    """Find most common words across all descriptions."""
    all_words = []
    for text in all_texts:
        words = text.lower().split()
        # Clean punctuation
        words = [''.join(c for c in word if c not in string.punctuation) for word in words]
        words = [word for word in words if word and len(word) > 2]  # Filter short words
        all_words.extend(words)
    
    return Counter(all_words).most_common(top_n)

def analyze_content_patterns(text):
    """Analyze content patterns in the text."""
    patterns = {
        'dialogue_indicators': len(re.findall(r'["\'].*?["\']', text)),
        'character_names': len(re.findall(r'\b[A-Z][a-z]+\b', text)),  # Capitalized words
        'action_words': len(re.findall(r'\b(runs?|walks?|looks?|turns?|moves?|goes?|comes?|enters?|exits?|sits?|stands?)\b', text, re.IGNORECASE)),
        'camera_directions': len(re.findall(r'\b(camera|shot|close|wide|pan|zoom|focus)\b', text, re.IGNORECASE)),
        'temporal_markers': len(re.findall(r'\b(then|now|later|before|after|meanwhile|suddenly|finally)\b', text, re.IGNORECASE))
    }
    return patterns

def analyze_ids(data):
    """Analyze the ID patterns."""
    ids = [int(key) for key in data.keys() if key.isdigit()]
    return {
        'min_id': min(ids) if ids else 0,
        'max_id': max(ids) if ids else 0,
        'id_range': max(ids) - min(ids) if ids else 0,
        'gaps_in_sequence': len(set(range(min(ids), max(ids) + 1)) - set(ids)) if ids else 0
    }

def main():
    try:
        # Load the JSON file
        with open('chatgpt.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("🎬 COMPREHENSIVE DATASET ANALYSIS")
        print("=" * 60)
        print(f"📁 Dataset: chatgpt.json")
        print(f"📊 Total entries: {len(data)}")
        print("=" * 60)
        
        # Initialize containers for statistics
        word_counts = []
        char_counts_total = []
        char_counts_no_spaces = []
        sentence_counts = []
        vocab_stats = []
        content_patterns = defaultdict(list)
        all_descriptions = list(data.values())
        
        # Analyze each description
        for key, description in data.items():
            # Basic counts
            words = len(description.split())
            chars_total, chars_no_spaces = count_characters(description)
            sentences = count_sentences(description)
            vocab = analyze_vocabulary(description)
            patterns = analyze_content_patterns(description)
            
            word_counts.append(words)
            char_counts_total.append(chars_total)
            char_counts_no_spaces.append(chars_no_spaces)
            sentence_counts.append(sentences)
            vocab_stats.append(vocab)
            
            for pattern_type, count in patterns.items():
                content_patterns[pattern_type].append(count)
        
        # Calculate statistics
        print("\n📝 TEXT LENGTH STATISTICS")
        print("-" * 40)
        print(f"Words per description:")
        print(f"  • Average: {statistics.mean(word_counts):.1f}")
        print(f"  • Median: {statistics.median(word_counts):.1f}")
        print(f"  • Min: {min(word_counts)}")
        print(f"  • Max: {max(word_counts)}")
        print(f"  • Std Dev: {statistics.stdev(word_counts):.1f}")
        
        print(f"\nCharacters per description:")
        print(f"  • Average: {statistics.mean(char_counts_total):.1f}")
        print(f"  • Average (no spaces): {statistics.mean(char_counts_no_spaces):.1f}")
        
        print(f"\nSentences per description:")
        print(f"  • Average: {statistics.mean(sentence_counts):.1f}")
        print(f"  • Min: {min(sentence_counts)}")
        print(f"  • Max: {max(sentence_counts)}")
        
        # Vocabulary analysis
        print("\n📚 VOCABULARY ANALYSIS")
        print("-" * 40)
        avg_unique_words = statistics.mean([v['unique_words'] for v in vocab_stats])
        avg_vocab_richness = statistics.mean([v['vocabulary_richness'] for v in vocab_stats])
        print(f"Average unique words per description: {avg_unique_words:.1f}")
        print(f"Average vocabulary richness: {avg_vocab_richness:.3f}")
        
        # Most common words
        print(f"\nMost common words (excluding short words):")
        common_words = get_common_words(all_descriptions, 15)
        for i, (word, count) in enumerate(common_words, 1):
            print(f"  {i:2}. {word:<12} ({count:,} times)")
        
        # Content pattern analysis
        print("\n🎭 CONTENT PATTERN ANALYSIS")
        print("-" * 40)
        for pattern_type, counts in content_patterns.items():
            avg_count = statistics.mean(counts)
            total_count = sum(counts)
            print(f"{pattern_type.replace('_', ' ').title()}:")
            print(f"  • Average per description: {avg_count:.1f}")
            print(f"  • Total across dataset: {total_count:,}")
        
        # ID analysis
        print("\n🔢 ID STRUCTURE ANALYSIS")
        print("-" * 40)
        id_stats = analyze_ids(data)
        print(f"ID range: {id_stats['min_id']} - {id_stats['max_id']}")
        print(f"Total range span: {id_stats['id_range']:,}")
        print(f"Missing IDs in sequence: {id_stats['gaps_in_sequence']:,}")
        print(f"Coverage: {(len(data) / (id_stats['id_range'] + 1) * 100):.1f}% of possible range")
        
        # Distribution analysis
        print("\n📈 DISTRIBUTION INSIGHTS")
        print("-" * 40)
        
        # Word count quartiles
        word_quartiles = [
            min(word_counts),
            statistics.quantiles(word_counts, n=4)[0],
            statistics.median(word_counts),
            statistics.quantiles(word_counts, n=4)[2],
            max(word_counts)
        ]
        print("Word count quartiles:")
        labels = ["Min", "Q1", "Median", "Q3", "Max"]
        for label, value in zip(labels, word_quartiles):
            print(f"  • {label}: {value:.0f}")
        
        # Find extremes
        print("\n🔍 EXTREME EXAMPLES")
        print("-" * 40)
        
        # Shortest description
        min_idx = word_counts.index(min(word_counts))
        min_key = list(data.keys())[min_idx]
        print(f"Shortest description (ID {min_key}): {min(word_counts)} words")
        print(f"Preview: {list(data.values())[min_idx][:100]}...")
        
        # Longest description
        max_idx = word_counts.index(max(word_counts))
        max_key = list(data.keys())[max_idx]
        print(f"\nLongest description (ID {max_key}): {max(word_counts)} words")
        print(f"Preview: {list(data.values())[max_idx][:100]}...")
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete!")
        
    except FileNotFoundError:
        print("❌ Error: chatgpt.json file not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import json
import statistics
from collections import Counter
import math

def calculate_mode(data):
    """Calculate mode(s) - most frequently occurring value(s)."""
    counter = Counter(data)
    max_count = max(counter.values())
    modes = [value for value, count in counter.items() if count == max_count]
    return modes, max_count

def calculate_percentiles(data, percentiles):
    """Calculate specific percentiles."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    result = {}
    
    for p in percentiles:
        if p == 0:
            result[p] = sorted_data[0]
        elif p == 100:
            result[p] = sorted_data[-1]
        else:
            # Using linear interpolation method
            index = (p / 100) * (n - 1)
            if index.is_integer():
                result[p] = sorted_data[int(index)]
            else:
                lower_index = int(index)
                upper_index = lower_index + 1
                weight = index - lower_index
                result[p] = sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    return result

def calculate_skewness(data):
    """Calculate skewness using the third moment."""
    n = len(data)
    mean_val = statistics.mean(data)
    std_val = statistics.stdev(data)
    
    # Third moment
    third_moment = sum((x - mean_val) ** 3 for x in data) / n
    
    # Skewness
    skewness = third_moment / (std_val ** 3)
    return skewness

def calculate_kurtosis(data):
    """Calculate kurtosis using the fourth moment."""
    n = len(data)
    mean_val = statistics.mean(data)
    std_val = statistics.stdev(data)
    
    # Fourth moment
    fourth_moment = sum((x - mean_val) ** 4 for x in data) / n
    
    # Kurtosis (excess kurtosis = kurtosis - 3)
    kurtosis = fourth_moment / (std_val ** 4)
    excess_kurtosis = kurtosis - 3
    return kurtosis, excess_kurtosis

def interpret_skewness(skew):
    """Interpret skewness value."""
    if abs(skew) < 0.5:
        return "approximately symmetric"
    elif skew < -0.5:
        return "left-skewed (negative skew)"
    else:
        return "right-skewed (positive skew)"

def interpret_kurtosis(excess_kurt):
    """Interpret excess kurtosis value."""
    if abs(excess_kurt) < 0.5:
        return "mesokurtic (normal-like)"
    elif excess_kurt < -0.5:
        return "platykurtic (flatter than normal)"
    else:
        return "leptokurtic (more peaked than normal)"

def main():
    try:
        # Load the JSON file
        with open('chatgpt.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract word counts
        word_counts = []
        for description in data.values():
            word_count = len(description.split())
            word_counts.append(word_count)
        
        print("📊 COMPREHENSIVE DESCRIPTIVE STATISTICS")
        print("=" * 60)
        print(f"Dataset: chatgpt.json")
        print(f"Sample size (n): {len(word_counts):,}")
        print("=" * 60)
        
        # MEASURES OF CENTRAL TENDENCY
        print("\n📍 MEASURES OF CENTRAL TENDENCY")
        print("-" * 40)
        
        mean_val = statistics.mean(word_counts)
        median_val = statistics.median(word_counts)
        modes, mode_frequency = calculate_mode(word_counts)
        
        print(f"Mean (μ):           {mean_val:.2f} words")
        print(f"Median:             {median_val:.1f} words")
        print(f"Mode(s):            {modes} (appears {mode_frequency} times)")
        
        if len(modes) == 1:
            mode_type = "Unimodal"
        elif len(modes) == 2:
            mode_type = "Bimodal"
        else:
            mode_type = f"Multimodal ({len(modes)} modes)"
        
        print(f"Distribution type:  {mode_type}")
        
        # MEASURES OF VARIABILITY/SPREAD
        print("\n📏 MEASURES OF VARIABILITY")
        print("-" * 40)
        
        variance_val = statistics.variance(word_counts)
        std_val = statistics.stdev(word_counts)
        range_val = max(word_counts) - min(word_counts)
        
        print(f"Range:              {range_val} words ({min(word_counts)} - {max(word_counts)})")
        print(f"Variance (σ²):      {variance_val:.2f}")
        print(f"Standard deviation (σ): {std_val:.2f} words")
        print(f"Coefficient of variation: {(std_val/mean_val)*100:.2f}%")
        
        # QUARTILES AND IQR
        print("\n📐 QUARTILES AND INTERQUARTILE RANGE")
        print("-" * 40)
        
        q1 = statistics.quantiles(word_counts, n=4)[0]
        q2 = statistics.median(word_counts)  # Same as median
        q3 = statistics.quantiles(word_counts, n=4)[2]
        iqr = q3 - q1
        
        print(f"Q1 (25th percentile): {q1:.1f} words")
        print(f"Q2 (50th percentile): {q2:.1f} words (median)")
        print(f"Q3 (75th percentile): {q3:.1f} words")
        print(f"IQR (Q3 - Q1):        {iqr:.1f} words")
        
        # PERCENTILES
        print("\n📊 KEY PERCENTILES")
        print("-" * 40)
        
        percentiles_to_calc = [5, 10, 25, 50, 75, 90, 95, 99]
        percentile_values = calculate_percentiles(word_counts, percentiles_to_calc)
        
        for p in percentiles_to_calc:
            print(f"P{p:2d}: {percentile_values[p]:6.1f} words")
        
        # SHAPE STATISTICS
        print("\n📈 DISTRIBUTION SHAPE")
        print("-" * 40)
        
        skewness_val = calculate_skewness(word_counts)
        kurtosis_val, excess_kurtosis_val = calculate_kurtosis(word_counts)
        
        print(f"Skewness:           {skewness_val:.3f} ({interpret_skewness(skewness_val)})")
        print(f"Kurtosis:           {kurtosis_val:.3f}")
        print(f"Excess kurtosis:    {excess_kurtosis_val:.3f} ({interpret_kurtosis(excess_kurtosis_val)})")
        
        # OUTLIER DETECTION
        print("\n🔍 OUTLIER ANALYSIS")
        print("-" * 40)
        
        # IQR method
        iqr_lower = q1 - 1.5 * iqr
        iqr_upper = q3 + 1.5 * iqr
        iqr_outliers = [x for x in word_counts if x < iqr_lower or x > iqr_upper]
        
        # Z-score method (>2 standard deviations)
        z_outliers = [x for x in word_counts if abs(x - mean_val) > 2 * std_val]
        
        print(f"IQR method outliers:  {len(iqr_outliers)} descriptions")
        print(f"  Threshold range:    {iqr_lower:.1f} - {iqr_upper:.1f}")
        
        print(f"Z-score outliers:     {len(z_outliers)} descriptions (>2σ)")
        print(f"  Threshold range:    {mean_val - 2*std_val:.1f} - {mean_val + 2*std_val:.1f}")
        
        # SUMMARY STATISTICS TABLE
        print("\n📋 SUMMARY STATISTICS TABLE")
        print("-" * 40)
        print(f"{'Statistic':<20} {'Value':<10} {'Unit'}")
        print("-" * 40)
        print(f"{'Count':<20} {len(word_counts):<10} {'descriptions'}")
        print(f"{'Mean':<20} {mean_val:<10.2f} {'words'}")
        print(f"{'Median':<20} {median_val:<10.1f} {'words'}")
        print(f"{'Mode':<20} {modes[0] if len(modes)==1 else 'Multiple':<10} {'words'}")
        print(f"{'Std Deviation':<20} {std_val:<10.2f} {'words'}")
        print(f"{'Variance':<20} {variance_val:<10.2f} {'words²'}")
        print(f"{'Range':<20} {range_val:<10} {'words'}")
        print(f"{'IQR':<20} {iqr:<10.1f} {'words'}")
        print(f"{'Skewness':<20} {skewness_val:<10.3f} {''}")
        print(f"{'Excess Kurtosis':<20} {excess_kurtosis_val:<10.3f} {''}")
        
        # NORMALITY ASSESSMENT
        print(f"\n🔬 NORMALITY ASSESSMENT")
        print("-" * 40)
        
        # Simple normality checks
        normal_indicators = []
        
        # Mean ≈ Median for normal distribution
        mean_median_diff = abs(mean_val - median_val)
        if mean_median_diff < std_val * 0.1:
            normal_indicators.append("✓ Mean ≈ Median")
        else:
            normal_indicators.append("✗ Mean ≠ Median")
        
        # Skewness close to 0
        if abs(skewness_val) < 0.5:
            normal_indicators.append("✓ Low skewness")
        else:
            normal_indicators.append("✗ High skewness")
        
        # Excess kurtosis close to 0
        if abs(excess_kurtosis_val) < 0.5:
            normal_indicators.append("✓ Normal kurtosis")
        else:
            normal_indicators.append("✗ Abnormal kurtosis")
        
        # 68-95-99.7 rule approximation
        within_1sd = len([x for x in word_counts if abs(x - mean_val) <= std_val]) / len(word_counts)
        within_2sd = len([x for x in word_counts if abs(x - mean_val) <= 2 * std_val]) / len(word_counts)
        
        if 0.65 <= within_1sd <= 0.71:
            normal_indicators.append("✓ ~68% within 1σ")
        else:
            normal_indicators.append(f"✗ {within_1sd:.1%} within 1σ (expected ~68%)")
            
        if 0.93 <= within_2sd <= 0.97:
            normal_indicators.append("✓ ~95% within 2σ")
        else:
            normal_indicators.append(f"✗ {within_2sd:.1%} within 2σ (expected ~95%)")
        
        for indicator in normal_indicators:
            print(f"  {indicator}")
        
        # Overall assessment
        positive_checks = sum(1 for ind in normal_indicators if ind.startswith("✓"))
        normality_score = positive_checks / len(normal_indicators)
        
        if normality_score >= 0.8:
            normality_assessment = "Very likely normal"
        elif normality_score >= 0.6:
            normality_assessment = "Approximately normal"
        else:
            normality_assessment = "Likely not normal"
            
        print(f"\nOverall assessment: {normality_assessment} ({positive_checks}/{len(normal_indicators)} checks passed)")
        
        print("\n" + "=" * 60)
        print("✅ Statistical analysis complete!")
        
    except FileNotFoundError:
        print("❌ Error: chatgpt.json file not found!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 
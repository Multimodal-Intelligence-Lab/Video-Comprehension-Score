from typing import Dict, List

def generate_summary_text(summary_data: Dict[str, float], segments: List[Dict]) -> str:
    summary_text = (
        f"Summary Statistics:\n"
        f"• Actual Line Length: {summary_data.get('actual_line_length', 0):.2f}\n"
        f"• Floor Ideal Length: {summary_data.get('floor_ideal_line_length', 0):.2f}\n"
        f"• Ceiling Ideal Length: {summary_data.get('ceil_ideal_line_length', 0):.2f}\n"
        f"• NAS-L Value: {summary_data.get('value', 0):.4f}\n"
        f"• Total Segments: {len(segments)}\n"
        f"• Calculable Segments: {sum(1 for s in segments if s.get('is_calculable', False))}\n"
    )
    return summary_text

def generate_calculation_method_text(summary_data: Dict[str, float]) -> str:
    actual_length = summary_data.get('actual_line_length', 0)
    floor_length = summary_data.get('floor_ideal_line_length', 0)
    ceil_length = summary_data.get('ceil_ideal_line_length', 0)
    nas_value = summary_data.get('value', 0)
    
    if actual_length <= ceil_length and actual_length >= floor_length:
        return f"• Calculation: Actual length is within ideal band → NAS-L = 1.0\n"
    elif actual_length < floor_length:
        return (f"• Calculation: Actual < Floor → NAS-L = {actual_length:.2f}"
                f" / {floor_length:.2f}"
                f" = {nas_value:.4f}\n")
    else:
        return (f"• Calculation: Actual > Ceiling → NAS-L = {ceil_length:.2f}"
                f" / {actual_length:.2f}"
                f" = {nas_value:.4f}\n")

def generate_lct_note(lct: int) -> str:
    if lct > 0:
        return f"\nNote: Local Chronology Tolerance (LCT) = {lct} is applied to extend the valid range for segment calculations.\n"
    return ""
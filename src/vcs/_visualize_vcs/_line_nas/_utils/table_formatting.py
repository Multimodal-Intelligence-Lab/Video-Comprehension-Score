from typing import Dict, List

def create_segment_table_header() -> str:
    """Create the header for the segment details table."""
    header_text = "Segment Details:\n\n"
    header_text += "┌" + "─" * 132 + "┐\n"
    header_text += "│ {:^5} │ {:^20} │ {:^20} │ {:^7} │ {:^7} │ {:^10} │ {:^11} │ {:^9} │ {:^15} │ {:^8} │\n".format(
        "Seg", "Start (x,y)", "End (x,y)", "dx", "dy", "Threshold", "LCT Thresh", "Calculable", "Method", "Length")
    header_text += "├" + "─" * 132 + "┤\n"
    return header_text

def create_segment_table_footer() -> str:
    """Create the footer for the segment details table."""
    return "└" + "─" * 132 + "┘\n"

def format_segment_row(segment: Dict, index: int, lct: int) -> str:
    """Format a single segment row for the table."""
    # Get segment data
    start = segment.get('start', (0, 0))
    end = segment.get('end', (0, 0))
    dx = segment.get('dx', 0)
    dy = segment.get('dy', 0)
    threshold = segment.get('threshold', 0)
    threshold_with_lct = segment.get('threshold_with_lct', 'N/A')
    is_calculable = segment.get('is_calculable', False)
    calc_method = segment.get('calculation_method', 'none')
    
    # Format method nicely
    method_map = {
        "standard": "Standard",
        "lct_capped": "LCT",
        "none": "N/A"
    }
    method = method_map.get(calc_method, "N/A")
    
    # Format the length
    length = segment.get('length', 0)
    length_str = f"{length:.2f}" if is_calculable else "0"
    
    # Format LCT threshold
    lct_thresh_str = f"{threshold_with_lct:.2f}" if threshold_with_lct != 'N/A' and lct > 0 else "N/A"
    
    # Create the row
    row = "│ {:^5} │ {:^20} │ {:^20} │ {:^7.2f} │ {:^7.2f} │ {:^10.2f} │ {:^11} │ {:^9} │ {:^15} │ {:^8} │\n".format(
        f"S{index+1}", 
        f"({start[0]},{start[1]})", 
        f"({end[0]},{end[1]})", 
        dx, dy, threshold, 
        lct_thresh_str,
        "Yes" if is_calculable else "No", 
        method, 
        length_str
    )
    
    return row

def create_segment_table(segments: List[Dict], lct: int) -> str:
    """Create complete segment table with header, rows, and footer."""
    table_text = create_segment_table_header()
    
    # Add rows for each segment
    for i, segment in enumerate(segments):
        table_text += format_segment_row(segment, i, lct)
    
    table_text += create_segment_table_footer()
    return table_text
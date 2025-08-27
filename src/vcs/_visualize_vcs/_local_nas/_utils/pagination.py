from typing import List, Dict

def paginate_segments(segments: List[Dict], max_per_page: int = 15) -> List[List[Dict]]:
    """Split segments into pages for better readability."""
    pages = []
    for i in range(0, len(segments), max_per_page):
        pages.append(segments[i:i+max_per_page])
    return pages

def should_paginate(segments: List[Dict], threshold: int = 15) -> bool:
    """Determine if segments should be paginated based on count."""
    return len(segments) > threshold

def calculate_total_pages(segments: List[Dict], max_per_page: int = 15) -> int:
    """Calculate total number of pages needed for segments."""
    if not segments:
        return 1
    return (len(segments) + max_per_page - 1) // max_per_page
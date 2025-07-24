from typing import List, Tuple

def split_text_chunks_for_display(chunks: List[str], max_per_page: int = 25) -> List[List[Tuple[int, str]]]:
    """Split chunks into pages with index information. Reduced from 40 to 25 for better text formatting."""
    pages = []
    for i in range(0, len(chunks), max_per_page):
        page_chunks = [(j, chunk) for j, chunk in enumerate(chunks[i:i+max_per_page], start=i+1)]
        pages.append(page_chunks)
    return pages

def should_paginate_chunks(chunks: List[str], threshold: int = 25) -> bool:
    """Determine if chunks should be paginated based on count."""
    return len(chunks) > threshold

def calculate_chunk_pages(chunks: List[str], max_per_page: int = 25) -> int:
    """Calculate total number of pages needed for chunks."""
    if not chunks:
        return 1
    return (len(chunks) + max_per_page - 1) // max_per_page

def get_page_range_info(chunks_with_indices: List[Tuple[int, str]]) -> Tuple[int, int]:
    """Get the range of chunk indices for a page."""
    if not chunks_with_indices:
        return 0, 0
    return chunks_with_indices[0][0], chunks_with_indices[-1][0]
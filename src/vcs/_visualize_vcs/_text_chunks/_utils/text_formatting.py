import textwrap
from typing import List, Tuple

def create_chunk_header(title: str, total_chunks: int, chunk_size: int, 
                       page_num: int = 1, total_pages: int = 1) -> str:
    """Create header section for chunk display."""
    header_text = f"{title.upper()}\n"
    header_text += "=" * 100 + "\n\n"
    
    # Add summary information
    header_text += f"Total Chunks: {total_chunks}\n"
    header_text += f"Chunk Size: {chunk_size}\n"
    if total_pages > 1:
        header_text += f"Page {page_num} of {total_pages}\n"
    header_text += "\n"
    header_text += "-" * 100 + "\n\n"
    
    return header_text

def wrap_chunk_text(chunk: str, width: int = 80, subsequent_indent: str = "     ") -> List[str]:
    """Wrap text chunk to fit within reasonable line length."""
    return textwrap.wrap(chunk, width=width, subsequent_indent=subsequent_indent)

def format_chunk_content(chunks_with_indices: List[Tuple[int, str]]) -> str:
    """Format chunk content with proper wrapping and spacing."""
    content_text = ""
    
    for i, (idx, chunk) in enumerate(chunks_with_indices):
        # Wrap long text chunks
        wrapped_lines = wrap_chunk_text(chunk)
        
        # Format each chunk with consistent spacing
        if len(wrapped_lines) == 1:
            # Single line chunk
            content_text += f"{idx:3d}. {wrapped_lines[0]}\n\n"
        else:
            # Multi-line chunk
            content_text += f"{idx:3d}. {wrapped_lines[0]}\n"
            for line in wrapped_lines[1:]:
                content_text += f"     {line}\n"
            content_text += "\n"
        
        # Add a subtle separator every 5 chunks for readability
        if (i + 1) % 5 == 0 and i < len(chunks_with_indices) - 1:
            content_text += create_chunk_separators()
    
    return content_text

def create_chunk_separators() -> str:
    """Create subtle separators between chunk groups."""
    return "     " + "·" * 90 + "\n\n"

def format_page_info(page_num: int, total_pages: int, start_chunk: int, 
                    end_chunk: int, total_chunks: int) -> str:
    """Format page information for multi-page displays."""
    if total_pages <= 1:
        return ""
    
    return f"\nPage {page_num} of {total_pages} | Showing chunks {start_chunk} to {end_chunk} of {total_chunks}"

def create_content_footer() -> str:
    """Create footer for chunk content."""
    return "=" * 100 + "\n"
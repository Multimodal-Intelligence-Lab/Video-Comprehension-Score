import numpy as np
import torch
from typing import List, Tuple, Callable

def _segment_and_chunk_texts(
    reference_text: str, 
    generated_text: str, 
    chunk_size: int,
    segmenter_fn: Callable
) -> Tuple[List[str], List[str]]:

    ref_segments = segmenter_fn(reference_text)
    gen_segments = segmenter_fn(generated_text)
        
    ref_chunks = _group_segments(ref_segments, chunk_size)
    gen_chunks = _group_segments(gen_segments, chunk_size)
        
    return ref_chunks, gen_chunks

def _group_segments(segments: List[str], chunk_size: int) -> List[str]:
    return [" ".join(segments[i:i + chunk_size]) for i in range(0, len(segments), chunk_size)]

def _build_similarity_matrix(
    ref_chunks: List[str],
    gen_chunks: List[str],
    embedding_fn: Callable
) -> Tuple[np.ndarray, int, int]:
    
    ref_tensor = embedding_fn(ref_chunks)
    gen_tensor = embedding_fn(gen_chunks)
    sim_matrix = torch.matmul(ref_tensor, gen_tensor.T).cpu().numpy()
    return sim_matrix, len(ref_chunks), len(gen_chunks)
import numpy as np
import torch
from typing import List, Tuple, Callable

from ._validation import _validate_embedding_output, _validate_segments

def _segment_and_chunk_texts(
    reference_text: str, 
    generated_text: str, 
    chunk_size: int,
    segmenter_fn: Callable[[str], List[str]]
) -> Tuple[List[str], List[str]]:

    ref_segments = segmenter_fn(reference_text)
    _validate_segments(ref_segments, "reference")
    gen_segments = segmenter_fn(generated_text)
    _validate_segments(gen_segments, "generated")

    ref_chunks = _group_segments(ref_segments, chunk_size)
    gen_chunks = _group_segments(gen_segments, chunk_size)
        
    return ref_chunks, gen_chunks

def _group_segments(segments: List[str], chunk_size: int) -> List[str]:
    return [" ".join(segments[i:i + chunk_size]) for i in range(0, len(segments), chunk_size)]

def _build_similarity_matrix(
    ref_chunks: List[str],
    gen_chunks: List[str],
    embedding_fn: Callable[[List[str]], torch.Tensor]
) -> Tuple[np.ndarray, int, int]:
    
    ref_tensor = _validate_embedding_output(
        embedding_fn(ref_chunks), len(ref_chunks), "embedding_fn"
    )
    gen_tensor = _validate_embedding_output(
        embedding_fn(gen_chunks), len(gen_chunks), "embedding_fn"
    )
    sim_matrix = _similarity_from_chunk_embeddings(ref_tensor, gen_tensor)
    return sim_matrix, len(ref_chunks), len(gen_chunks)


def _similarity_from_chunk_embeddings(
    ref_tensor: torch.Tensor,
    gen_tensor: torch.Tensor,
) -> np.ndarray:
    """Dot-product similarity matrix, shape (ref_len, gen_len), clamped at
    0: anti-correlated chunks carry no more alignment signal than unrelated
    ones, and a negative similarity must not act as LESS-than-zero evidence
    anywhere downstream (load-penalty coverage, F1 inputs). The single
    shared site for this op so both entry points are bit-identical.
    Doc-level GAS is deliberately NOT clamped (it feeds the SAS gate)."""
    return torch.matmul(ref_tensor, gen_tensor.T).clamp_min(0).cpu().numpy()
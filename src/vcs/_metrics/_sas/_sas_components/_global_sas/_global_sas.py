import torch
import torch.nn.functional as F
from typing import Callable

def _compute_gas_metrics(
    reference_text: str, 
    generated_text: str, 
    embedding_fn: Callable
) -> float:

    emb_all = embedding_fn([reference_text, generated_text])
    if len(emb_all) < 2:
        return 0.0
    
    ref_vec = emb_all[0].unsqueeze(0)
    gen_vec = emb_all[1].unsqueeze(0)
    sim = F.cosine_similarity(ref_vec, gen_vec, dim=1)
    
    return sim.item()
"""Deterministic offline embedders and segmenters for the VCS test suite.

No network, no model downloads: embeddings are hashed character-trigram
bags, L2-normalized, float64. hashlib.md5 is used (not the built-in
``hash()``, which is salted per process) so vectors are stable across
runs, machines, and Python versions.
"""
import hashlib

import numpy as np
import torch


def _trigram_vector(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    padded = f"##{text.lower()}##"
    for i in range(len(padded) - 2):
        digest = hashlib.md5(padded[i:i + 3].encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        # Empty (or fully sign-cancelling) text: fall back to a fixed unit
        # vector so rows always satisfy the unit-norm embedding contract.
        vec[0] = 1.0
        return vec
    return vec / norm


def make_embedder(dim: int):
    def embed(texts):
        if len(texts) == 0:
            return torch.from_numpy(np.zeros((0, dim), dtype=np.float64))
        mat = np.stack([_trigram_vector(t, dim) for t in texts])
        return torch.from_numpy(mat)

    return embed


embed_dim64 = make_embedder(64)
embed_dim48 = make_embedder(48)


def split_sentences(text: str):
    return [s.strip() for s in text.split(".") if s.strip()]


def split_keep_empty(text: str):
    """Period split WITHOUT filtering: trailing '.' yields an empty-string
    segment, exercising the degenerate empty-chunk path."""
    return text.split(".")

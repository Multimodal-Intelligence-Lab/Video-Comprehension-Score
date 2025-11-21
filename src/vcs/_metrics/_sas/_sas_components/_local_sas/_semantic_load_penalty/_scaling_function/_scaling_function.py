def _apply_scaling_function(similarity: float, semantic_coverage: float) -> float:
    """
    Apply the affine scaling function based on semantic coverage.
    
    scaling_function(s) = {
        0,                                          if coverage=0 or s ≤ 1-coverage
        (s - (1-coverage)) / coverage,             if coverage∈(0,1] and s > 1-coverage
    }
    
    Key properties:
    - scaling_function(1) = 1 (perfect similarity maps to 1)
    - Threshold at 1-coverage
    - Monotone and bounded in [0,1]
    - When coverage=1: scaling_function(s) = s (no penalty when adequacy is perfect)
    
    Args:
        similarity: Original similarity value
        semantic_coverage: Semantic coverage ratio (0 ≤ coverage ≤ 1)
    
    Returns:
        Scaled similarity value based on semantic coverage
    """
    if semantic_coverage == 0 or similarity <= (1 - semantic_coverage):
        return 0.0
    else:
        return (similarity - (1 - semantic_coverage)) / semantic_coverage
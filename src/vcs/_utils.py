def _calculate_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom else 0.0

def _compute_sas(gas: float, las: float) -> float:
    if las <= 0:
        return 0.0
    val = gas - (1 - las)
    return (val / las) if (val > 0) else 0.0

def _compute_vcs_scaled(gas_scaled: float, nas: float) -> float:
    if gas_scaled < nas:
        numerator = gas_scaled - (1 - nas)
        denominator = nas
    else:
        numerator = nas - (1 - gas_scaled)
        denominator = gas_scaled

    return (numerator / denominator) if (numerator > 0 and denominator != 0) else 0.0

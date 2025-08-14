import math

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def safe_log1m(x: float, eps: float = 1e-12) -> float:
    x = clamp01(x)
    return math.log(max(eps, 1.0 - x))

def solve_alpha0(i_p: float, wp: float, L0: float) -> float:
    # (1 - i_p)^alpha0 * (1 - wp)^(1 - alpha0) = 1 - L0
    Li = safe_log1m(i_p)
    Lw = safe_log1m(wp)
    Ll = safe_log1m(L0)
    den = (Li - Lw)
    if abs(den) < 1e-12:
        return 0.5
    return clamp01((Ll - Lw) / den)
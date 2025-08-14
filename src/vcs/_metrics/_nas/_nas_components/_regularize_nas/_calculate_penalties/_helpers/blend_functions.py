from .math_utils import clamp01, solve_alpha0

def blend_noisy_or(i_p: float, wp: float, k: float) -> float:
    """
    L_p = 1 - (1 - i_p)^alpha * (1 - wp)^(1 - alpha)
    Choose alpha so k=0 reproduces L0 = M^2 + m(1 - M),
    k=+1 => alpha=0 => L_p=wp, k=-1 => alpha=1 => L_p=i_p.
    """
    i_p = clamp01(i_p)
    wp = clamp01(wp)
    k = max(-1.0, min(1.0, k))

    M, m = (i_p, wp) if i_p >= wp else (wp, i_p)
    L0 = M * M + m * (1.0 - M)
    if L0 >= 1.0 - 1e-12:
        return 1.0  # saturated

    alpha0 = solve_alpha0(i_p, wp, L0)

    if k >= 0.0:  # push toward wp
        alpha = (1.0 - k) * alpha0          # k=+1 => alpha=0
    else:          # push toward i_p
        alpha = alpha0 + (1.0 - alpha0) * (-k)  # k=-1 => alpha=1

    alpha = clamp01(alpha)
    beta = 1.0 - alpha
    L_p = 1.0 - (1.0 - i_p) ** alpha * (1.0 - wp) ** beta
    return clamp01(L_p)
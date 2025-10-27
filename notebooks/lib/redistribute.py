import numpy as np
from .utils import normalize_l1

def redistribute_confidences_of_class(
    conf: np.ndarray,               # (N, C)
    proj_conf: np. ndarray,         # (N, C)
    avg_proj_conf: np.ndarray,      # (C, )
    removed_class: int,
) -> np.ndarray:
    
    # Make Mask
    _, C = conf.shape
    keep_mask = np.ones(C, dtype=bool)
    keep_mask[removed_class] = False

    # v_forget
    r2 = normalize_l1(avg_proj_conf[keep_mask])         # (C-1, )
    left_hand = proj_conf[:, removed_class][:, None] * r2  # (N, 1) * (C-1,) -> (N, C-1)

    # v_retain
    eps = 1e-12
    current_ratio = (
        (1 - proj_conf[:, removed_class]) / 
        (1 - conf[:, removed_class] + eps) 
    )       # (N, ) * (N,) -> (N,)
    right_hand = current_ratio[:, None] * conf[:, keep_mask] # (N, 1) * (N, C-1) -> (N, C-1)

    res = left_hand + right_hand # (N, C-1)

    # === Apply Redistribution ===
    result = np.zeros_like(conf)
    result[:, keep_mask] =  normalize_l1(res, axis=1) # (N, C-1)

    return result # (N, C)



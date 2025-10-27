# Projection Pack
import numpy as np
from numpy.linalg import norm
from scipy.linalg import null_space
from .utils import normalize_l1

def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """Orthonormalize a set of vectors using Gram-Schmidt."""
    vectors = np.array(vectors, dtype=float)
    ortho = []
    for v in vectors:
        for u in ortho:
            v -= np.dot(v, u) * u
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-10:
            v /= norm_v
            ortho.append(v)
    return np.array(ortho)

def project_confidences(
        class_confidences: dict, 
        avg_confidences_ortho: np.ndarray, 
        removed_class_idx: int
) -> dict:
    """
    Project confidence vectors into orthogonal plane removing influence of one class.
    class_confidences: dict[class_idx -> list of (N,C) arrays]
    avg_confidences_ortho: (C,C) matrix after Gram-Schmidt
    removed_class_idx: int

    Output is norm_l1(abs(proj))
    """

    avg_conf_removed = avg_confidences_ortho[removed_class_idx]

    # Find projection plane
    as_unit = avg_conf_removed / norm(avg_conf_removed)
    as_row = as_unit.reshape(1, -1)
    ortho_plane = null_space(as_row)

    proj_confidences = {}
    for cls, conf_list in class_confidences.items():
        conf_cls = np.stack(conf_list)          # (N,C)
        proj = conf_cls @ ortho_plane @ ortho_plane.T
        abs_proj = np.abs(proj)
        norm_proj = normalize_l1(abs_proj, axis=1)
        proj_confidences[cls] = norm_proj
    return proj_confidences
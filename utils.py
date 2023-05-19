import os

import numpy as np
from numpy import ndarray


def brute_force_matching(features_1: ndarray, features_2: ndarray) -> float:
    dot_product = features_1 @ features_2.T
    
    # best matches
    indices_1 = dot_product.argmax(axis=1)
    indices_2 = dot_product.argmax(axis=0)
    
    indices_i = indices_1 if len(indices_1) < len(indices_2) else indices_2
    indices_j = indices_2 if len(indices_1) < len(indices_2) else indices_1
    
    num_matches = (indices_j[indices_i] == np.arange(0, len(indices_i))).sum()
    return num_matches


def cosine_similarity(features_1: ndarray, features_2: ndarray) -> float:
    norm_1 = np.linalg.norm(features_1)
    norm_2 = np.linalg.norm(features_2)
    cosine = (features_1 @ features_2) / (norm_1 * norm_2)
    return cosine


def jpg_filename(filename: str) -> str:
    basename, _ = os.path.splitext(filename)
    return f"{basename}.jpg"

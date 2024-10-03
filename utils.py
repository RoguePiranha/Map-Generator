import numpy as np


def get_neighbors(x, y, terrain):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            if 0 <= x + i < terrain.shape[0] and 0 <= y + j < terrain.shape[1]:
                neighbors.append((x + i, y + j))
    return neighbors


def normalize(array):
    """Normalize an array to range 0-1."""
    min_val = np.min(array)
    max_val = np.max(array)

    # prevent division by zero
    if max_val == min_val:
        return np.zeros_like(array)

    return (array - min_val) / (max_val - min_val)

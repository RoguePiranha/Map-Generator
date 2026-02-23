from utils import get_neighbors
import random
import numpy as np
from scipy.ndimage import gaussian_filter1d


def add_rivers(heightmap, terrain, config):
    """Add rivers that flow from mountain springs down to lakes/ocean."""
    num_rivers = config.get("NUM_RIVERS", 15)
    river_width = config.get("RIVER_WIDTH", 4)
    water_threshold = config["WATER_THRESHOLD"]
    mountain_threshold = config["MOUNTAIN_THRESHOLD"]

    rows, cols = heightmap.shape

    # Find "spring" candidates: cells near the mountain/plains boundary
    spring_min = config.get("RIVER_THRESHOLD", 0.55)
    spring_max = mountain_threshold + 0.05
    mask = (heightmap >= spring_min) & (heightmap <= spring_max)
    candidates = list(zip(*np.where(mask)))

    if not candidates:
        return terrain

    random.shuffle(candidates)
    rivers_placed = 0
    river_starts = []  # Track start positions to space rivers apart
    min_dist_between_starts = 150

    for sx, sy in candidates:
        if rivers_placed >= num_rivers:
            break

        # Don't start too close to an existing river source
        too_close = False
        for ux, uy in river_starts:
            if abs(sx - ux) + abs(sy - uy) < min_dist_between_starts:
                too_close = True
                break
        if too_close:
            continue

        path = _trace_river_downhill(sx, sy, heightmap, terrain, water_threshold)

        if len(path) >= 80:  # Only keep reasonably long rivers
            smoothed = _smooth_path(path, sigma=6)
            _draw_wide_river(terrain, smoothed, river_width)
            river_starts.append((sx, sy))
            rivers_placed += 1

    return terrain


def _trace_river_downhill(x, y, heightmap, terrain, water_threshold):
    """Trace a river from (x,y) downhill toward water, tolerating small flat areas."""
    rows, cols = heightmap.shape
    path = [(x, y)]
    visited = set()
    visited.add((x, y))

    max_steps = 2000
    flat_budget = 30  # Allow crossing some flat/slightly-uphill cells
    flat_used = 0

    for _ in range(max_steps):
        curr_h = heightmap[x][y]

        # Reached a water body — success
        if curr_h < water_threshold or terrain[x][y] in (0, 6):
            break

        # Look at all 8 neighbors
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                    neighbors.append((nx, ny))

        if not neighbors:
            break

        # Pick the lowest neighbor
        best = min(neighbors, key=lambda p: heightmap[p[0]][p[1]])
        bx, by = best
        best_h = heightmap[bx][by]

        if best_h < curr_h:
            # Strictly downhill — always accept
            x, y = bx, by
            path.append((x, y))
            visited.add((x, y))
        elif flat_used < flat_budget and (best_h - curr_h) < 0.02:
            # Flat or tiny uphill — spend budget to cross it
            flat_used += 1
            x, y = bx, by
            path.append((x, y))
            visited.add((x, y))
        else:
            break

    return path


def _smooth_path(path, sigma=6):
    """Smooth a list of (x,y) coordinates with a Gaussian filter for natural curves."""
    if len(path) < 4:
        return path
    arr = np.array(path, dtype=float)
    arr[:, 0] = gaussian_filter1d(arr[:, 0], sigma=sigma)
    arr[:, 1] = gaussian_filter1d(arr[:, 1], sigma=sigma)
    # Round back to integer grid coords and deduplicate
    smoothed = []
    prev = None
    for row in arr:
        pt = (int(round(row[0])), int(round(row[1])))
        if pt != prev:
            smoothed.append(pt)
            prev = pt
    return smoothed


def _draw_wide_river(terrain, path, half_width):
    """Draw a river path with width, tapering from narrow at source to wider downstream."""
    rows, cols = terrain.shape
    length = len(path)
    for idx, (px, py) in enumerate(path):
        progress = idx / max(length - 1, 1)
        current_hw = max(1, int(half_width * (0.3 + 0.7 * progress)))
        for dx in range(-current_hw, current_hw + 1):
            for dy in range(-current_hw, current_hw + 1):
                if dx * dx + dy * dy <= current_hw * current_hw:
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        if terrain[nx][ny] not in (3, 4):
                            terrain[nx][ny] = 5


def add_lakes(terrain, heightmap, config):
    """Mark low-elevation water cells as lakes (vectorized)."""
    mask = (heightmap < config["WATER_THRESHOLD"]) & (terrain != 5)
    terrain[mask] = 6  # Lake
    return terrain


def add_ponds(terrain, config):
    """Randomly place ponds on plains or forests (vectorized)."""
    eligible = (terrain == 1) | (terrain == 2)
    random_vals = np.random.random(terrain.shape)
    terrain[eligible & (random_vals < config["POND_PROBABILITY"])] = 7  # Pond
    return terrain


def add_caves(terrain, heightmap, config):
    """Randomly place caves in elevated terrain (vectorized)."""
    eligible = heightmap > config["PLAINS_THRESHOLD"]
    random_vals = np.random.random(terrain.shape)
    terrain[eligible & (random_vals < config["CAVE_PROBABILITY"])] = 9  # Cave
    return terrain

import numpy as np
import noise
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathfinding import a_star
from terrain import add_rivers, add_lakes, add_ponds, add_caves
from utils import get_neighbors, normalize
from placeVillages import place_villages
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from color_map import create_gradient_color_map


def smooth_heightmap(heightmap, sigma=1):
    """Apply Gaussian smoothing to the heightmap."""
    return gaussian_filter(heightmap, sigma=sigma)


def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed, sigma=2):
    heightmap = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            x = i / scale
            y = j / scale
            heightmap[i][j] = noise.pnoise2(
                x,
                y,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=width,
                repeaty=height,
                base=seed,
            )

    # Apply Gaussian smoothing
    heightmap = smooth_heightmap(heightmap, sigma=sigma)
    return heightmap


def visualize_map_with_features(heightmap, terrain, cmap, terrain_colors):
    """Visualize the map with the gradient color map for heightmap and overlay discrete terrain features."""
    plt.figure(figsize=(10, 10))

    # Show the heightmap with gradient color map (vmin/vmax ensure colorbar matches 0-1)
    plt.imshow(heightmap, cmap=cmap, vmin=0.0, vmax=1.0)

    # Create masked overlay - only show discrete features (villages, roads, rivers, etc.)
    # Base terrain (water, forest, plains, mountains) is rendered via the gradient colormap
    feature_mask = (terrain >= 4)  # Only overlay features: villages, rivers, lakes, ponds, roads, caves
    overlay = np.full_like(terrain, fill_value=np.nan)
    overlay[feature_mask] = terrain[feature_mask]

    # Build a colormap mapping terrain IDs 4-9 to their colors
    feature_ids = [4, 5, 6, 7, 8, 9]
    feature_color_list = [terrain_colors[fid] for fid in feature_ids]
    cmap_overlay = mcolors.ListedColormap(feature_color_list)
    bounds = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    norm_overlay = mcolors.BoundaryNorm(bounds, cmap_overlay.N)
    plt.imshow(overlay, cmap=cmap_overlay, norm=norm_overlay, interpolation='nearest', alpha=0.9)

    # Add contour lines for the heightmap
    plt.contour(heightmap, levels=10, colors="black", linewidths=0.3, alpha=0.4)

    # Create a legend for the discrete features with labels
    legend_labels = {
        0: 'Water',
        1: 'Forest',
        2: 'Plains',
        3: 'Mountains',
        4: 'Villages',
        5: 'Rivers',
        6: 'Lakes',
        7: 'Ponds',
        8: 'Roads',
        9: 'Caves',
    }

    legend_patches = [
        mpatches.Patch(color=terrain_colors[i], label=legend_labels[i])
        for i in legend_labels
    ]
    plt.legend(handles=legend_patches, loc="upper right", fontsize="small")

    plt.colorbar()
    plt.title("Generated Terrain Map with Features Overlay")
    plt.show()


def generate_terrain(heightmap, config):
    terrain = np.zeros_like(heightmap)  # Default: 0 (Water)
    terrain[heightmap > config["WATER_THRESHOLD"]] = 1    # Forest
    terrain[heightmap > config["PLAINS_THRESHOLD"]] = 2   # Plains
    terrain[heightmap > config["MOUNTAIN_THRESHOLD"]] = 3  # Mountain
    return terrain


def add_roads(terrain, villages, road_width=2, max_road_distance=1200, extra_connections=0):
    """Connect villages with a sparse, natural-looking road network.

    Uses a minimum spanning tree (MST) so every village is reachable while
    avoiding dense over-connected road meshes.
    """
    if len(villages) < 2:
        return terrain

    road_edges = _build_mst_edges(villages)

    # Optional extra short links for slight redundancy (kept low by default)
    if extra_connections > 0:
        road_edges.extend(_pick_extra_edges(villages, road_edges, extra_connections, max_road_distance))

    for i, j in road_edges:
        x1, y1 = villages[i]
        x2, y2 = villages[j]
        road_path = a_star((x1, y1), (x2, y2), terrain)
        if len(road_path) < 2:
            continue
        smoothed = _smooth_road(road_path)
        _widen_path(terrain, smoothed, road_width, value=8, blocked={3})
    return terrain


def _build_mst_edges(villages):
    """Build a minimum spanning tree of village centers using Euclidean distance."""
    n = len(villages)
    in_tree = [False] * n
    in_tree[0] = True
    edges = []

    while len(edges) < n - 1:
        best = None
        best_dist = float("inf")

        for i in range(n):
            if not in_tree[i]:
                continue
            xi, yi = villages[i]
            for j in range(n):
                if in_tree[j]:
                    continue
                xj, yj = villages[j]
                dist = np.hypot(xi - xj, yi - yj)
                if dist < best_dist:
                    best_dist = dist
                    best = (i, j)

        if best is None:
            break

        i, j = best
        edges.append((i, j))
        in_tree[j] = True

    return edges


def _pick_extra_edges(villages, existing_edges, extra_connections, max_road_distance):
    """Pick a few short non-MST edges for realism without over-densifying roads."""
    n = len(villages)
    existing = {tuple(sorted(edge)) for edge in existing_edges}
    candidates = []

    for i in range(n):
        xi, yi = villages[i]
        for j in range(i + 1, n):
            edge = (i, j)
            if edge in existing:
                continue
            xj, yj = villages[j]
            dist = np.hypot(xi - xj, yi - yj)
            if dist <= max_road_distance:
                candidates.append((dist, edge))

    candidates.sort(key=lambda item: item[0])
    return [edge for _, edge in candidates[:extra_connections]]


def _smooth_road(path):
    """Smooth an A* grid path into a natural curve without spline loops."""
    if len(path) < 4:
        return path

    arr = np.array(path, dtype=float)
    # Coordinate smoothing keeps the route topology while removing stair-steps.
    sigma = max(1.0, min(4.0, len(arr) / 70.0))
    arr[:, 0] = gaussian_filter1d(arr[:, 0], sigma=sigma)
    arr[:, 1] = gaussian_filter1d(arr[:, 1], sigma=sigma)

    # Convert back to integer grid coords, deduplicate
    smoothed = []
    prev = None
    for row in arr:
        pt = (int(round(row[0])), int(round(row[1])))
        if pt != prev:
            smoothed.append(pt)
            prev = pt
    return smoothed


def _widen_path(terrain, path, half_width, value=8, blocked=None):
    """Draw a path onto the terrain grid with a given pixel half-width."""
    if blocked is None:
        blocked = set()
    rows, cols = terrain.shape
    for px, py in path:
        for dx in range(-half_width, half_width + 1):
            for dy in range(-half_width, half_width + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < rows and 0 <= ny < cols and terrain[nx][ny] not in blocked:
                    terrain[nx][ny] = value


def run_map_generation(config, terrain_colors):
    # Create heightmap
    heightmap = generate_heightmap(
        config["WIDTH"],
        config["HEIGHT"],
        config["SCALE"],
        config["OCTAVES"],
        config["PERSISTENCE"],
        config["LACUNARITY"],
        config["SEED"],
        sigma=config.get("SMOOTHING_SIGMA", 2),
    )
    heightmap = normalize(heightmap)
    print("Heightmap generated and normalized")

    # Generate terrain and features
    terrain = generate_terrain(heightmap, config)
    print("Terrain generated")
    terrain = add_rivers(heightmap, terrain, config)
    print("Rivers added")
    terrain = add_lakes(terrain, heightmap, config)
    print("Lakes added")
    terrain = add_ponds(terrain, config)
    print("Ponds added")
    terrain = add_caves(terrain, heightmap, config)
    print("Caves added")

    # Place villages and roads
    villages = place_villages(terrain, config["NUM_VILLAGES"], config["VILLAGE_RADIUS"])
    terrain = add_roads(
        terrain,
        villages,
        road_width=config.get("ROAD_WIDTH", 2),
        max_road_distance=config.get("ROAD_MAX_DISTANCE", 1200),
        extra_connections=config.get("ROAD_EXTRA_CONNECTIONS", 0),
    )
    print("Villages and roads placed")

    # Visualize
    cmap = create_gradient_color_map()
    # norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalizing for smooth gradients
    print("Visualizing Map Externally...")
    visualize_map_with_features(heightmap, terrain, cmap, terrain_colors)
    print("Map closed. Thank you for using the Map Generator!")

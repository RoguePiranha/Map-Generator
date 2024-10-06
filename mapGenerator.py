import numpy as np
import noise
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from pathfinding import a_star
from terrain import add_rivers, add_lakes, add_ponds, add_caves
from utils import get_neighbors, normalize
from placeVillages import place_villages
from scipy.ndimage import gaussian_filter
from color_map import create_gradient_color_map


def smooth_heightmap(heightmap, sigma=1):
    """Apply Gaussian smoothing to the heightmap."""
    return gaussian_filter(heightmap, sigma=sigma)


def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed):
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
                repeatx=1024,
                repeaty=1024,
                base=seed,
            )

    # Apply Gaussian smoothing
    heightmap = smooth_heightmap(heightmap, sigma=2)
    return heightmap


def visualize_map_with_features(heightmap, terrain, cmap, terrain_colors):
    """Visualize the map with the gradient color map for heightmap and overlay discrete terrain features."""
    plt.figure(figsize=(10, 10))

    # Show the heightmap with gradient color map
    plt.imshow(heightmap, cmap=cmap)

    # Create a copy of the terrain for feature overlay
    overlay = np.zeros_like(terrain)

    # Overlay discrete terrain features (like roads, rivers, villages, etc.)
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if terrain[i][j] in terrain_colors:
                overlay[i][j] = terrain[i][j]  # Mark discrete feature in the overlay grid

    # Show the overlay using terrain_colors
    cmap_overlay = mcolors.ListedColormap([terrain_colors[i] for i in range(len(terrain_colors))])
    plt.imshow(overlay, cmap=cmap_overlay, alpha=0.6)  # Overlay with some transparency

    # Add contour lines for the heightmap
    plt.contour(heightmap, levels=10, colors="black", linewidths=0.5)

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
    terrain = np.zeros_like(heightmap)
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            if heightmap[i][j] > config["MOUNTAIN_THRESHOLD"]:
                terrain[i][j] = 3  # Mountain
            elif heightmap[i][j] > config["PLAINS_THRESHOLD"]:
                terrain[i][j] = 2  # Plains
            elif heightmap[i][j] > config["WATER_THRESHOLD"]:
                terrain[i][j] = 1  # Forest
            else:
                terrain[i][j] = 0  # Water (lake or river)
    return terrain


def add_roads(terrain, villages):
    for i in range(len(villages)):
        for j in range(i + 1, len(villages)):
            x1, y1 = villages[i]
            x2, y2 = villages[j]
            road_path = a_star((x1, y1), (x2, y2), terrain)
            for x, y in road_path:
                if terrain[x][y] != 3:  # Avoid marking roads on mountains
                    terrain[x][y] = 8  # Mark road
    return terrain


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
    terrain = add_roads(terrain, villages)
    print("Villages and roads placed")

    # Visualize
    cmap = create_gradient_color_map()
    # norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalizing for smooth gradients
    print("Visualizing Map Externally...")
    visualize_map_with_features(heightmap, terrain, cmap, terrain_colors)
    print("Map closed. Thank you for using the Map Generator!")

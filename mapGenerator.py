import numpy as np
import noise
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathfinding import a_star
from terrain import add_rivers, add_lakes, add_ponds, add_caves
from utils import get_neighbors, normalize
from placeVillages import place_villages, village_radius


def create_color_map(terrain_colors):
    cmap = mcolors.ListedColormap([terrain_colors[i] for i in range(10)])
    bounds = list(range(11))  # Boundaries between colors
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


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
    return heightmap


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


# Function to add cliffs and canyons
def add_cliffs_and_canyons(heightmap, terrain, config):
    gradient_threshold = config[
        "CLIFF_THRESHOLD"
    ]  # Adjust this value for steeper cliffs
    for i in range(1, heightmap.shape[0] - 1):
        for j in range(1, heightmap.shape[1] - 1):
            current_height = heightmap[i][j]
            neighbors = get_neighbors(i, j, heightmap)

            # Calculate the maximum height difference between the current cell and its neighbors
            max_height_diff = max(
                abs(current_height - heightmap[x, y]) for x, y in neighbors
            )

            # Mark as cliff if the height difference exceeds the threshold
            if max_height_diff > gradient_threshold:
                terrain[i][j] = 10  # Mark as cliff

            # If it's a lowland area and near a river or steep side, mark it as a canyon
            if (
                current_height < config["WATER_THRESHOLD"]
                and max_height_diff > gradient_threshold
            ):
                terrain[i][j] = 11  # Mark as canyon
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


def visualize_map(terrain, cmap, norm):
    plt.figure(figsize=(10, 10))
    plt.imshow(terrain, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.title("Generated Terrain Map with Custom Colors")
    plt.show()


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
    print("Heightmap generated")

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

    # Add cliffs and canyons
    terrain = add_cliffs_and_canyons(heightmap, terrain, config)
    print("Cliffs and canyons added")

    # Place villages and roads
    villages = place_villages(
        terrain, config["NUM_VILLAGES"], village_radius(config["VILLAGE_RADIUS"])
    )
    terrain = add_roads(terrain, villages)
    print("Villages and roads placed")

    # Visualize
    cmap, norm = create_color_map(terrain_colors)
    print("Visualizing map...")
    visualize_map(terrain, cmap, norm)
    print("Map generation complete.")

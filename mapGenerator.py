import numpy as np
import noise
import matplotlib.pyplot as plt
import random
from pathfinding import a_star, get_neighbors  # Import the A* pathfinding function

# Constants
WIDTH = 512
HEIGHT = 512
SCALE = 100.0  # Controls how zoomed in/out the noise is
OCTAVES = 6
PERSISTENCE = 0.5
LACUNARITY = 2.0
SEED = 42  # Ensures reproducibility

# Terrain type thresholds
MOUNTAIN_THRESHOLD = 0.6
PLAINS_THRESHOLD = 0.4
WATER_THRESHOLD = 0.3

# Feature constants
RIVER_THRESHOLD = 0.25  # Minimum height for a river to start
NUM_VILLAGES = 5
POND_PROBABILITY = 0.01  # 1% chance of placing a pond on plains or forests
CAVE_PROBABILITY = 0.02  # 2% chance of placing a cave in mountains/hills

# Generate Perlin noise-based heightmap
def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed):
    heightmap = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            x = i / scale
            y = j / scale
            heightmap[i][j] = noise.pnoise2(x, y, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)
    return heightmap

# Normalize heightmap to range 0-1
def normalize(array):
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)

# Generate the terrain map based on thresholds
def generate_terrain(heightmap):
    terrain = np.zeros_like(heightmap)
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            if heightmap[i][j] > MOUNTAIN_THRESHOLD:
                terrain[i][j] = 3  # Mountain
            elif heightmap[i][j] > PLAINS_THRESHOLD:
                terrain[i][j] = 2  # Plains
            elif heightmap[i][j] > WATER_THRESHOLD:
                terrain[i][j] = 1  # Forest
            else:
                terrain[i][j] = 0  # Water (lake or river)
    return terrain

# Function to add rivers to the terrain
def add_rivers(heightmap, terrain):
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if heightmap[i][j] > RIVER_THRESHOLD and heightmap[i][j] < PLAINS_THRESHOLD:
                trace_river(i, j, heightmap, terrain)
    return terrain

def trace_river(x, y, heightmap, terrain):
    """ Simulate a river flowing downhill by tracing through lower points. """
    while True:
        # Current point is now a river
        terrain[x][y] = 5  # River

        # Get the lowest adjacent point to flow to
        neighbors = get_neighbors(x, y, heightmap)
        if not neighbors:
            break
        next_x, next_y = min(neighbors, key=lambda pos: heightmap[pos[0], pos[1]])

        # If no lower neighbor, stop the river
        if heightmap[next_x, next_y] >= heightmap[x, y]:
            break

        # Move to the next position
        x, y = next_x, next_y

# Place lakes in low-lying, non-river areas
def add_lakes(terrain, heightmap):
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if heightmap[i][j] < WATER_THRESHOLD and terrain[i][j] != 5:  # Not already a river
                terrain[i][j] = 6  # Lake
    return terrain

# Add ponds randomly in suitable locations (forests, plains)
def add_ponds(terrain):
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if terrain[i][j] in (1, 2) and random.random() < POND_PROBABILITY:  # In forest/plains
                terrain[i][j] = 7  # Pond
    return terrain

# Add caves to mountains and hills (elevated areas)
def add_caves(terrain, heightmap):
    for i in range(WIDTH):
        for j in range(HEIGHT):
            # Caves more likely in mountains and hills
            if heightmap[i][j] > PLAINS_THRESHOLD and random.random() < CAVE_PROBABILITY:
                terrain[i][j] = 9  # Cave
    return terrain

# Place villages on the map (near plains)
def place_villages(terrain, num_villages):
    village_positions = []
    for _ in range(num_villages):
        while True:
            x = random.randint(0, terrain.shape[0] - 1)
            y = random.randint(0, terrain.shape[1] - 1)
            if terrain[x][y] == 2:  # Only place villages on plains
                village_positions.append((x, y))
                terrain[x][y] = 4  # Mark village
                break
    return village_positions

# Draw roads between villages using A* pathfinding
def add_roads(terrain, villages):
    for i in range(len(villages)):
        for j in range(i + 1, len(villages)):
            x1, y1 = villages[i]
            x2, y2 = villages[j]
            road_path = a_star((x1, y1), (x2, y2), terrain)
            for (x, y) in road_path:
                if terrain[x][y] != 3:  # Avoid marking roads on mountains
                    terrain[x][y] = 8  # Mark road
    return terrain

# Visualize the map
def visualize_map(terrain):
    plt.figure(figsize=(10, 10))
    plt.imshow(terrain, cmap='terrain')
    plt.colorbar()
    plt.title('Generated Terrain Map')
    plt.show()

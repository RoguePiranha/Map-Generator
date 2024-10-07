'''
This version of the map generator is designed to combine multiple
generation methods to create a more realistic and detailed terrain map.

The heightmap is generated using Perlin noise and Diamond-Square algorithm.
The terrain is classified into water, forest, plains, and mountains.
Villages are placed on plains and connected by roads using A* pathfinding.
Rivers are traced from mountains and lakes are added in low elevation areas.

Torch has been added in order to make use of the GPU for faster computation.
The map generation process is visualized using matplotlib.
'''


import numpy as np
import noise
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
import heapq

from placeVillages import expand_village_organically

from tqdm import tqdm
import torch


#########################################################################################
#                                    INDEX                                              #
#
#   1. Configuration
#       ├── Config
#       └── Terrain Colors
#
#   2. Create Gradient Color Map
#   3. Place Villages
#       ├── Place Villages
#       ├── Village Radius Randomizer
#       └── Expand Village Organically
#
#   4. A* Pathfinding Algorithm
#       ├── Terrain Costs
#       └── A* Pathfinding
#
#   5. Utils
#       ├── Get Neighbors
#       └── Normalize
#
#   6. Map Generator
#      ├── Generate Perlin Noise
#      ├── Diamond Square
#      ├── Apply Erosion
#      ├── Generate Heightmap
#      └── Generate Terrain and Features
#
#   7. Visualize Map with Features
#   8. Run Map Generation
#
#########################################################################################



#########################################################################################
# Function Set: Configuration
# Variables: config, terrain_colors
#
# config: Dictionary containing adjustable parameters for map generation.
# terrain_colors: Dictionary mapping terrain types to colors for visualization.
#########################################################################################

config = {
    "WIDTH": 512,                # Width of the map
    "HEIGHT": 512,               # Height of the map
    "SCALE": 100.0,              # Scale of the Perlin noise
    "OCTAVES": 6,                # Number of octaves for Perlin noise
    "PERSISTENCE": 0.5,          # Persistence for Perlin noise
    "LACUNARITY": 2.0,           # Lacunarity for Perlin noise
    "SEED": 42,                  # Seed for Perlin noise
    "ROUGHNESS": 0.7,            # Roughness for Diamond-Square algorithm
    "MOUNTAIN_THRESHOLD": 0.65,  # Elevation threshold for mountains
    "PLAINS_THRESHOLD": 0.45,    # Elevation threshold for plains
    "WATER_THRESHOLD": 0.3,      # Elevation threshold for water bodies
    "NUM_VILLAGES": 10,          # Number of villages to place
    "VILLAGE_RADIUS": 5,         # Radius for village expansion
}

terrain_colors = {
    0: 'blue',           # Water
    1: 'darkgreen',      # Forest
    2: 'lightgreen',     # Plains
    3: 'brown',          # Mountains
    4: 'yellow',         # Villages
    5: 'deepskyblue',    # Rivers
    6: 'cyan',           # Lakes
    7: 'lightblue',      # Ponds
    8: 'black',          # Roads
}

#########################################################################################
# END OF CONFIGURATION
#########################################################################################


#########################################################################################
# Function Name: create_gradient_color_map
# Input: None
# Output: cmap (matplotlib.colors.LinearSegmentedColormap)
# Logic:
#     1. This function creates a gradient color map for the terrain.
#     2. It uses the LinearSegmentedColormap class from the matplotlib.colors module.
#     3. The color map is defined with specific colors for different terrain types.
#     4. The color transitions are defined to ensure natural transitions between terrains.
#     5. The color map is returned as cmap.
# Example call: cmap = create_gradient_color_map()
#########################################################################################

def create_gradient_color_map():
    from matplotlib import colors

    # Adjusted color map to ensure natural transitions and include more detailed features
    cmap = colors.LinearSegmentedColormap.from_list(
        'terrain_map', 
        [
            (0.0, "darkblue"),   # Deep water
            (0.2, "blue"),       # Shallow water
            (0.25, "cyan"),      # Lakes and water bodies
            (0.3, "beige"),      # Beach/sand
            (0.4, "lightgreen"), # Plains
            (0.5, "yellowgreen"), # Low hills
            (0.6, "green"),      # Forest
            (0.70, "darkgreen"), # Dense forest
            (0.80, "saddlebrown"), # Mountain slopes
            (0.85, "gray"),      # Rocky mountain peaks
            (1.0, "white")       # Snowy mountain peaks
        ]
    )
    return cmap

#########################################################################################
# END OF FUNCTION
#########################################################################################


#########################################################################################
# Function Set: placeVillages
# Functions: place_villages, village_radius_randomizer, expand_village_organically, get_neighbors
#########################################################################################

def place_villages(terrain, num_villages, village_radius=3, village_size=100):
    village_positions = []

    for _ in range(num_villages):
        retries = 0
        while retries < 100:
            x = random.randint(village_radius, terrain.shape[0] - village_radius - 1)
            y = random.randint(village_radius, terrain.shape[1] - village_radius - 1)
            
            # Ensure no overlap
            if not any(abs(v[0] - x) <= village_radius and abs(v[1] - y) <= village_radius for v in village_positions):
                if terrain[x][y] == 2:  # Only place on plains
                    village_positions.append((x, y))
                    expand_village_organically(terrain, x, y, village_size)
                    break
            retries += 1
    return village_positions

def village_radius_randomizer(village_radius):
    """Randomize village radius."""
    return random.randint(village_radius // 2, village_radius * 2)

def expand_village_organically(terrain, x, y, max_village_size):
    """Expand the village organically with random flood-fill to avoid circular shapes."""
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    village_cells = 0

    while queue and village_cells < max_village_size:
        cx, cy = queue.popleft()

        # Mark the current cell as part of the village
        if terrain[cx][cy] == 2:  # Only expand into plains (value 2)
            terrain[cx][cy] = 4  # Mark as village
            village_cells += 1

        # Check neighboring cells and grow the village randomly within the area
        for nx, ny in get_neighbors(cx, cy, terrain):
            if (nx, ny) not in visited and terrain[nx][ny] == 2:
                visited.add((nx, ny))
                queue.append((nx, ny))



#########################################################################################
# END OF FUNCTION SET
#########################################################################################



#########################################################################################
# Function Name: a_star
# Input: start (tuple), goal (tuple), terrain (numpy.ndarray)
# Output: path (list of tuples)
# Logic:
#     1. This function implements the A* pathfinding algorithm to find the shortest path
#        from the start position to the goal position on the terrain map.
#     2. The terrain map is represented as a 2D numpy array with different terrain types.
#     3. The movement costs for different terrain types are defined in the TERRAIN_COSTS dictionary.
#     4. The algorithm uses a priority queue to explore the neighboring cells based on the cost.
#     5. The path is reconstructed by backtracking from the goal position to the start position.
#     6. The final path is returned as a list of tuples representing the coordinates.
# Example call: path = a_star((0, 0), (10, 10), terrain_map)
#########################################################################################

# Define movement costs for different terrains
TERRAIN_COSTS = {
    0: 100000000000000,    # Water (impassable or very expensive)
    1: 3,       # Forest (medium difficulty)
    2: 1,       # Plains (easy to travel)
    3: 100,     # Mountain (impassable or very expensive)
    4: 1,       # Village (destination)
    5: 10,       # River (can be crossed, but slightly more costly)
    6: 100000,     # Lake (impassable or very expensive)
    7: 20,      # Pond (slightly higher cost)
    8: 1,       # Road (already established, easy to follow)
    9: 10000000,    # Caves (impassable or very expensive)
    10: 10000000,   # Cliffs (impassable or very expensive)
    11: 100000,   # Canyons (impassable or very expensive)
}


# A* pathfinding algorithm
def a_star(start, goal, terrain):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

    open_set = []
    heapq.heappush(open_set, (0, start))  # (priority, (x, y))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for neighbor in get_neighbors(*current, terrain):
            # Ensure terrain values are integers and handle unexpected terrain values
            try:
                terrain_type = int(terrain[neighbor[0]][neighbor[1]])
                if terrain_type not in TERRAIN_COSTS:
                    raise ValueError(f"Unexpected terrain type: {terrain_type}")
            except ValueError:
                terrain_type = (
                    100  # Assign a default cost if the terrain type is invalid
                )

            # Calculate new cost for this path
            new_cost = cost_so_far[current] + TERRAIN_COSTS.get(terrain_type, 100)

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(goal, neighbor)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct the path
    path = []
    if goal in came_from:
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

    return path

#########################################################################################
# END OF FUNCTION
#########################################################################################


#########################################################################################
# Function Set: utils
# Functions: get_neighbors, normalize
#
# Function Name: get_neighbors
# Input: x (int), y (int), terrain (numpy.ndarray)
# Output: neighbors (list of tuples)
# 
# Function Name: normalize
# Input: array (numpy.ndarray)
# Output: normalized_array (numpy.ndarray)
#########################################################################################

def get_neighbors(x, y, terrain):
    neighbors = []
    # Get valid neighboring coordinates (8 directions: up, down, left, right, and diagonals)
    for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        if 0 <= x + i < terrain.shape[0] and 0 <= y + j < terrain.shape[1]:
            neighbors.append((x + i, y + j))
    return neighbors


def normalize(array):
    """Normalize an array to range 0-1."""
    min_val = torch.min(array)
    max_val = torch.max(array)
    if max_val == min_val:
        return torch.zeros_like(array)
    return (array - min_val) / (max_val - min_val)

#########################################################################################
# END OF FUNCTION SET
#########################################################################################


#########################################################################################
# Function Set: mapGenerator
# Functions: generate_perlin_noise, diamond_square, apply_erosion, generate_heightmap, generate_terrain_and_features
#########################################################################################

#########################################################################################
# Function Name: generate_perlin_noise
# Input: width (int), height (int), scale (float), octaves (int), persistence (float), lacunarity (float), seed (int)
# Output: heightmap (torch.Tensor)
# Logic:
#     1. This function generates a heightmap using Perlin noise.
#     2. The Perlin noise is generated using the noise library.
#     3. The heightmap is created by combining multiple octaves of Perlin noise.
#     4. The persistence and lacunarity parameters control the smoothness and detail of the noise.
#     5. The seed parameter is used to generate different noise patterns.
#     6. The heightmap is returned as a torch.Tensor after normalization.
# Example call: heightmap = generate_perlin_noise(512, 512, 100.0, 6, 0.5, 2.0, 42)
#########################################################################################

def generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed):
    """Generate base heightmap using Perlin noise."""
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, width, device='cuda') / scale,
        torch.arange(0, height, device='cuda') / scale
    )
    heightmap = torch.zeros((width, height), device='cuda')
    for octave in range(octaves):
        heightmap += (persistence ** octave) * torch.tensor([
            [noise.pnoise2(x.item(), y.item(), octaves=octave, persistence=persistence, lacunarity=lacunarity, base=seed) 
            for y in grid_y] for x in grid_x], device='cuda')
    torch.cuda.empty_cache()
    return normalize(heightmap)


#########################################################################################
# Function Name: diamond_square
# Input: heightmap (torch.Tensor), size (int), roughness (float)
# Output: heightmap (torch.Tensor)
# Logic:
#     1. This function generates a heightmap using the Diamond-Square algorithm.
#     2. The algorithm iteratively performs the diamond and square steps to create the terrain.
#     3. The roughness parameter controls the variation in elevation.
#     4. The heightmap is returned after the generation process.
# Example call: heightmap = diamond_square(heightmap, 511, 0.7)
#
# Note: The size parameter should be one less than the width or height of the heightmap.
#########################################################################################

def diamond_square(heightmap, size, roughness):
    """Generate heightmap using the Diamond-Square algorithm."""
    step_size = size
    half_step = step_size // 2
    max_height = 1.0

    while step_size > 1:
        # Diamond step
        for x in tqdm(range(0, size, step_size), desc="Diamond Step", unit="steps"):
            for y in range(0, size, step_size):
                if x + half_step < heightmap.shape[0] and y + half_step < heightmap.shape[1]:
                    avg = (
                        heightmap[x, y]
                        + heightmap[(x + step_size) % size, y]
                        + heightmap[x, (y + step_size) % size]
                        + heightmap[(x + step_size) % size, (y + step_size) % size]
                    ) / 4.0
                    heightmap[x + half_step, y + half_step] = avg + random.uniform(-max_height, max_height) * roughness

        # Square step
        for x in tqdm(range(0, size, half_step), desc="Square Step", unit="steps"):
            for y in range((x + half_step) % step_size, size, step_size):
                if x < heightmap.shape[0] and y < heightmap.shape[1]:
                    avg = (
                        heightmap[(x - half_step) % size, y]
                        + heightmap[(x + half_step) % size, y]
                        + heightmap[x, (y + half_step) % size]
                        + heightmap[x, (y - half_step) % size]
                    ) / 4.0
                    heightmap[x, y] = avg + random.uniform(-max_height, max_height) * roughness

        step_size //= 2
        half_step = step_size // 2
        max_height *= roughness

    torch.cuda.empty_cache()
    return heightmap


#########################################################################################
# Function Name: apply_erosion
# Input: heightmap (torch.Tensor), iterations (int)
# Output: heightmap (torch.Tensor)
# Logic:
#     1. This function applies an erosion simulation to the heightmap.
#     2. The erosion process helps in creating realistic river valleys and drainage patterns.
#     3. The algorithm iteratively updates the heightmap based on the neighboring cells.
#     4. The heightmap is returned after the erosion process.
# Example call: heightmap = apply_erosion(heightmap, 10)
#########################################################################################

def apply_erosion(heightmap, iterations=10):
    """Apply erosion simulation to add realistic river valleys and drainage patterns."""
    for _ in tqdm(range(iterations), desc="Applying Erosion", unit="iterations"):
        for i in range(1, heightmap.shape[0] - 1):
            for j in range(1, heightmap.shape[1] - 1):
                neighbors = [
                    heightmap[i - 1, j],
                    heightmap[i + 1, j],
                    heightmap[i, j - 1],
                    heightmap[i, j + 1],
                ]
                heightmap[i, j] -= 0.1 * (heightmap[i, j] - min(neighbors))
    torch.cuda.empty_cache()
    return heightmap


#########################################################################################
# Function Name: generate_heightmap
# Input: width (int), height (int), scale (float), octaves (int), persistence (float), lacunarity (float), seed (int), roughness (float)
# Output: heightmap (torch.Tensor)
# Logic:
#     1. This function generates a heightmap using a combination of Perlin noise and Diamond-Square algorithm.
#     2. The Perlin noise is used to create the base elevation of the terrain.
#     3. The Diamond-Square algorithm is used to add elevation variation and details.
#     4. The heightmap is normalized and smoothed using Gaussian filtering.
#     5. The final heightmap is returned after the generation process.
# Example call: heightmap = generate_heightmap(512, 512, 100.0, 6, 0.5, 2.0, 42, 0.7)
#
# Note: The width and height should be powers of 2 + 1 for the Diamond-Square algorithm.
#########################################################################################

def generate_heightmap(width, height, scale, octaves, persistence, lacunarity, seed, roughness):
    """Generate a heightmap using a combination of Perlin noise and Diamond-Square algorithm."""
    # Base elevation using Perlin noise
    heightmap = generate_perlin_noise(width, height, scale, octaves, persistence, lacunarity, seed)

    # Elevation variation using Diamond-Square
    diamond_square(heightmap, min(width, height) - 1, roughness)

    # Apply erosion for realism
    heightmap = apply_erosion(heightmap)

    # Normalize and smooth heightmap
    heightmap = normalize(heightmap)
    heightmap = gaussian_filter(heightmap.cpu().numpy(), sigma=2)
    heightmap = torch.tensor(heightmap, device='cuda')

    return heightmap


#########################################################################################
# Function Name: generate_terrain_and_features
# Input: heightmap (torch.Tensor), config (dict)
# Output: terrain (torch.Tensor)
# Logic:
#     1. This function generates terrain and adds features like rivers, lakes, villages, and roads in one pass.
#     2. The terrain is classified into water, forest, plains, and mountains based on the elevation.
#     3. Villages are placed on plains and expanded organically using a flood-fill algorithm.
#     4. Roads are added between villages using A* pathfinding.
#     5. The final terrain map with features is returned.
# Example call: terrain = generate_terrain_and_features(heightmap, config)
#########################################################################################

def generate_terrain_and_features(heightmap, config):
    """Generate terrain and add features like rivers, lakes, villages, and roads in one pass."""
    width, height = heightmap.shape
    terrain = torch.zeros((width, height), device='cuda')
    village_positions = []

    for i in tqdm(range(width), desc="Generating Terrain Features", unit="rows"):
        for j in range(height):
            elevation = heightmap[i][j]

            # Classify terrain
            if elevation > config["MOUNTAIN_THRESHOLD"]:
                terrain[i][j] = 3  # Mountain
            elif elevation > config["PLAINS_THRESHOLD"]:
                terrain[i][j] = 2  # Plains
            elif elevation > config["WATER_THRESHOLD"]:
                terrain[i][j] = 1  # Forest
            else:
                terrain[i][j] = 0  # Water

            # Place villages
            if terrain[i][j] == 2 and len(village_positions) < config["NUM_VILLAGES"] and random.random() < 0.005:
                village_positions.append((i, j))
                expand_village_organically(terrain, i, j, config["VILLAGE_RADIUS"])

    # Add roads using A* pathfinding
    for i in tqdm(range(len(village_positions)), desc="Adding Roads", unit="roads"):
        for j in range(i + 1, len(village_positions)):
            x1, y1 = village_positions[i]
            x2, y2 = village_positions[j]
            road_path = a_star((x1, y1), (x2, y2), terrain.cpu().numpy())
            for x, y in road_path:
                if terrain[x, y] != 3:  # Avoid marking roads on mountains
                    terrain[x, y] = 8  # Mark road

    return terrain

#########################################################################################
# END OF FUNCTION SET
#########################################################################################



#########################################################################################
# Function Name: visualize_map_with_features
# Input: heightmap (torch.Tensor), terrain (torch.Tensor), cmap (matplotlib.colors.LinearSegmentedColormap), terrain_colors (dict)
# Output: None
# Logic:
#     1. This function visualizes the map with the gradient color map for the heightmap.
#     2. It overlays discrete terrain features like water, forest, plains, mountains, villages, rivers, ponds, caves, and roads.
#     3. The terrain features are displayed using different colors based on the terrain_colors dictionary.
#     4. The heightmap is shown with contour lines and a colorbar for elevation.
#     5. The discrete features are displayed with a legend for easy identification.
#     6. The map is displayed using matplotlib.
# Example call: visualize_map_with_features(heightmap, terrain, cmap, terrain_colors)
#########################################################################################

def visualize_map_with_features(heightmap, terrain, cmap, terrain_colors):
    """Visualize the map with the gradient color map for heightmap and overlay discrete terrain features."""
    plt.figure(figsize=(10, 10))

    # Show the heightmap with gradient color map
    img = plt.imshow(heightmap.cpu(), cmap=cmap, interpolation='nearest')

    # Create a copy of the terrain for feature overlay
    overlay = terrain.cpu().numpy()

    # Create a ListedColormap based on terrain_colors values
    cmap_overlay = mcolors.ListedColormap([terrain_colors[i] for i in sorted(terrain_colors.keys())])

    # Show the overlay using terrain_colors
    plt.imshow(overlay, cmap=cmap_overlay, alpha=0.8)

    # Add contour lines for the heightmap
    plt.contour(heightmap.cpu(), levels=10, colors="black", linewidths=0.5)

    # Create a colorbar for the heightmap
    plt.colorbar(img)

    # Create a legend for the discrete features with labels
    legend_labels = {
        0: 'Water',
        1: 'Forest',
        2: 'Plains',
        3: 'Mountains',
        4: 'Villages',
        5: 'Rivers',
        6: 'Ponds',
        7: 'Caves',
        8: 'Roads'
    }

    legend_patches = [
        mpatches.Patch(color=terrain_colors[key], label=legend_labels[key])
        for key in legend_labels
    ]

    plt.legend(handles=legend_patches, loc="upper right", fontsize="small")
    plt.title("Generated Terrain Map with Features Overlay")
    plt.show()

#########################################################################################
# END OF FUNCTION
#########################################################################################


#########################################################################################
# Function Name: run_map_generation
# Input: config (dict), terrain_colors (dict)
# Output: None
# Logic:
#     1. This function runs the map generation process with the given configuration and terrain colors.
#     2. It generates the heightmap, terrain, and features in one pass.
#     3. The color map is created for visualization using the create_gradient_color_map function.
#     4. The map is visualized externally using the visualize_map_with_features function.
#     5. The user can view the generated map with terrain features and close the window.
# Example call: run_map_generation(config, terrain_colors)
#########################################################################################

def run_map_generation(config, terrain_colors):
    
    print("Generating Heightmap...")
    # Generate heightmap
    heightmap = generate_heightmap(
        config["WIDTH"],
        config["HEIGHT"],
        config["SCALE"],
        config["OCTAVES"],
        config["PERSISTENCE"],
        config["LACUNARITY"],
        config["SEED"],
        config["ROUGHNESS"]
    )
    print("Heightmap generated and normalized")

    print("Generating Terrain with Features...")
    # Generate terrain and features in one pass
    terrain = generate_terrain_and_features(heightmap, config)
    print("Terrain with features generated")

    # Correct color map creation
    cmap = create_gradient_color_map()

    print("Visualizing Map Externally...")
    visualize_map_with_features(heightmap, terrain, cmap, terrain_colors)
    print("Map closed. Thank you for using the Map Generator!")
    
#########################################################################################
# END OF FUNCTION
#########################################################################################



# Run the map generation
if __name__ == "__main__":
    run_map_generation(config, terrain_colors)
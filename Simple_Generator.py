import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import random

from tqdm import tqdm

# Map dimensions and parameters
width, height = 100, 100
ocean_level = 0.3  # Lower this for more ocean coverage
forest_density = 0.3

# Define color palettes
colors = {
    "deep_ocean": "#2B65EC",
    "ocean": "#4169E1",
    "shore": "#87CEFA",
    "plain": "#98FB98",
    "forest": "#2E8B57",
    "mountain": "#A9A9A9",
    "snow": "#FFFFFF",
    "river": "#1E90FF",
    "road": "#8B4513",
    "village": "#D2691E",
    "city": "#8B0000",
}


def generate_perlin_noise():
    """Generate Perlin-like noise using Gaussian blur for smooth transitions"""
    base_noise = np.random.rand(width, height)
    smoothed_noise = gaussian_filter(base_noise, sigma=6)
    return smoothed_noise


def apply_terrain(elevation):
    """Assign biomes and terrain based on elevation"""
    terrain = np.zeros((width, height, 3))
    

    # Apply colors based on height
    for x in tqdm(range(width)):
        
        for y in tqdm(range(height)):
            if elevation[x, y] < ocean_level * 0.6:
                terrain[x, y] = [43 / 255, 101 / 255, 236 / 255]  # Deep ocean
            elif elevation[x, y] < ocean_level:
                terrain[x, y] = [65 / 255, 105 / 255, 225 / 255]  # Ocean
            elif elevation[x, y] < ocean_level + 0.05:
                terrain[x, y] = [135 / 255, 206 / 255, 250 / 255]  # Shore
            elif elevation[x, y] < ocean_level + 0.3:
                terrain[x, y] = [152 / 255, 251 / 255, 152 / 255]  # Plains
            elif elevation[x, y] < ocean_level + 0.5:
                terrain[x, y] = [46 / 255, 139 / 255, 87 / 255]  # Forest
            elif elevation[x, y] < ocean_level + 0.7:
                terrain[x, y] = [169 / 255, 169 / 255, 169 / 255]  # Mountain
            else:
                terrain[x, y] = [255 / 255, 255 / 255, 255 / 255]  # Snow peaks
    return terrain


def add_rivers(elevation):
    """Generate rivers by tracing paths from higher elevations to lower ones"""
    river_map = np.zeros_like(elevation)
    num_rivers = 5
    for _ in tqdm(range(num_rivers)):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        # Start from high elevation
        while elevation[x, y] < ocean_level + 0.3:
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)

        # Trace river down to ocean
        for _ in range(100):
            river_map[x, y] = 1
            delta_x, delta_y = random.choice([-1, 0, 1]), random.choice([-1, 0, 1])
            x, y = min(max(0, x + delta_x), width - 1), min(
                max(0, y + delta_y), height - 1
            )
            if elevation[x, y] < ocean_level:
                break
    return river_map


def add_cities_and_villages(elevation, terrain):
    """Add cities and villages based on terrain type"""
    cities, villages = [], []
    for _ in range(3):  # Number of cities
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        if terrain[x, y][0] == 152 / 255:  # Cities on plains
            cities.append((x, y))
    for _ in range(8):  # Number of villages
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        if terrain[x, y][0] == 152 / 255:  # Villages on plains
            villages.append((x, y))
    return cities, villages


def plot_map(terrain, rivers, cities, villages):
    plt.figure(figsize=(10, 10))
    plt.imshow(terrain, extent=(0, width, 0, height))

    # Draw rivers
    for i in tqdm(range(width)):
        for j in range(height):
            if rivers[i, j] == 1:
                plt.plot(j, i, color=colors["river"], marker="s", markersize=2)

    # Draw cities and villages
    for x, y in cities:
        plt.plot(y, x, color=colors["city"], marker="o", markersize=8)
    for x, y in villages:
        plt.plot(y, x, color=colors["village"], marker="o", markersize=5)

    # Finalize map
    plt.axis("off")
    plt.show()


# Generate map components
elevation = generate_perlin_noise()
terrain = apply_terrain(elevation)
rivers = add_rivers(elevation)
cities, villages = add_cities_and_villages(elevation, terrain)

# Plot the map
plot_map(terrain, rivers, cities, villages)

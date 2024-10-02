from mapGenerator import run_map_generation

# Configuration dictionary for adjustable parameters
config = {
    "WIDTH": 512,
    "HEIGHT": 512,
    "SCALE": 100.0,           # Controls how zoomed in/out the noise is
    "OCTAVES": 6,
    "PERSISTENCE": 0.5,
    "LACUNARITY": 2.0,
    "SEED": 42,               # Ensures reproducibility
    "MOUNTAIN_THRESHOLD": 0.6,
    "PLAINS_THRESHOLD": 0.4,
    "WATER_THRESHOLD": 0.3,
    "RIVER_THRESHOLD": 0.25,  # Minimum height for a river to start
    "NUM_VILLAGES": 5,
    "POND_PROBABILITY": 0.01,  # 1% chance of placing a pond on plains or forests
    "CAVE_PROBABILITY": 0.02   # 2% chance of placing a cave in mountains/hills
}

# Custom color map for different terrain types
terrain_colors = {
    0: 'blue',      # Water (lakes and rivers)
    1: 'darkgreen', # Forest
    2: 'lightgreen',# Plains
    3: 'brown',     # Mountains
    4: 'yellow',    # Villages
    5: 'blue',      # Rivers (water)
    6: 'cyan',      # Lakes
    7: 'lightblue', # Ponds
    8: 'black',     # Roads
    9: 'gray',      # Caves
}

# Run the map generation
if __name__ == "__main__":
    run_map_generation(config, terrain_colors)

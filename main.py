from mapGenerator import run_map_generation

# Configuration dictionary for adjustable parameters
config = {
    "WIDTH": 2048,
    "HEIGHT": 2048,
    "SCALE": 500.0,             # Larger scales produce larger, smoother features
    "OCTAVES": 4,               # Fewer octaves = less high-frequency noise
    "PERSISTENCE": 0.35,        # Lower = smoother transitions between octaves
    "LACUNARITY": 2.0,
    "SMOOTHING_SIGMA": 6,       # Gaussian smoothing radius for the heightmap
    "SEED": 42,                 # Ensures reproducibility
    "MOUNTAIN_THRESHOLD": 0.72,
    "PLAINS_THRESHOLD": 0.45,
    "WATER_THRESHOLD": 0.28,
    "RIVER_THRESHOLD": 0.55,     # Minimum height for a river spring (high foothills)
    "RIVER_WIDTH": 5,           # Half-width of rivers at mouth in pixels
    "NUM_RIVERS": 12,           # Number of rivers to generate
    "ROAD_WIDTH": 2,            # Half-width of roads in pixels
    "ROAD_MAX_DISTANCE": 1200,  # Max distance for optional extra road links
    "ROAD_EXTRA_CONNECTIONS": 0,# Extra links beyond the minimum spanning network
    "NUM_VILLAGES": 10,
    "VILLAGE_RADIUS": 10,       # Sets average size of villages
    "POND_PROBABILITY": 0.0001, # % chance of placing a pond on plains or forests
    "CAVE_PROBABILITY": 0.0005, # % chance of placing a cave in mountains/hills
    "CLIFF_THRESHOLD": 0.05     # Adjust this for steep cliffs
}

terrain_colors = {
    0: 'blue',           # Water (lakes and rivers)
    1: 'darkgreen',      # Forest
    2: 'lightgreen',     # Plains
    3: 'brown',          # Mountains
    4: 'yellow',         # Villages
    5: 'deepskyblue',    # Rivers (water)
    6: 'cyan',           # Lakes
    7: 'lightblue',      # Ponds
    8: 'black',          # Roads
    9: 'gray',           # Caves
}


# Run the map generation
if __name__ == "__main__":
    run_map_generation(config, terrain_colors)
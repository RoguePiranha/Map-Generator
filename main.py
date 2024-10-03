from mapGenerator import run_map_generation

# Configuration dictionary for adjustable parameters
config = {
    "WIDTH": 2048,
    "HEIGHT": 2048,
    "SCALE": 200.0,             # Larger scales produce larger features (broad, slowly changing terrain)
    "OCTAVES": 6,
    "PERSISTENCE": 0.5,
    "LACUNARITY": 2.0,
    "SEED": 42,                 # Ensures reproducibility
    "MOUNTAIN_THRESHOLD": 0.6,
    "PLAINS_THRESHOLD": 0.4,
    "WATER_THRESHOLD": 0.3,
    "RIVER_THRESHOLD": 0.25,    # Minimum height for a river to start
    "NUM_VILLAGES": 15,
    "VILLAGE_RADIUS": 10,
    "POND_PROBABILITY": 0.0001,   # % chance of placing a pond on plains or forests
    "CAVE_PROBABILITY": 0.0005,  # % chance of placing a cave in mountains/hills
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
    10: 'saddlebrown',   # Cliffs
    11: 'darkred'        # Canyons
}


# Run the map generation
if __name__ == "__main__":
    run_map_generation(config, terrain_colors)
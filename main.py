import mapGenerator as mg

from mapGenerator import WIDTH, HEIGHT, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED, NUM_VILLAGES

# Main function
def generate_map():
    heightmap = mg.generate_heightmap(WIDTH, HEIGHT, SCALE, OCTAVES, PERSISTENCE, LACUNARITY, SEED)
    heightmap = mg.normalize(heightmap)
    terrain = mg.generate_terrain(heightmap)

    # Add rivers
    terrain = mg.add_rivers(heightmap, terrain)

    # Add lakes
    terrain = mg.add_lakes(terrain, heightmap)

    # Add ponds
    terrain = mg.add_ponds(terrain)

    # Add caves to mountains and hills
    terrain = mg.add_caves(terrain, heightmap)

    # Place villages
    villages = mg.place_villages(terrain, NUM_VILLAGES)

    # Add roads between villages using A* pathfinding
    terrain = mg.add_roads(terrain, villages)

    # Visualize the terrain with villages, roads, rivers, lakes, ponds, and caves
    mg.visualize_map(terrain)

# Call the map generation function
generate_map()

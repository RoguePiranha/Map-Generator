from utils import get_neighbors
import random


def add_rivers(heightmap, terrain, config):
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if (
                heightmap[i][j] > config["RIVER_THRESHOLD"]
                and heightmap[i][j] < config["PLAINS_THRESHOLD"]
            ):
                trace_river(i, j, heightmap, terrain)
    return terrain


def trace_river(x, y, heightmap, terrain):
    """Simulate a river flowing downhill by tracing through lower points."""
    while True:
        terrain[x][y] = 5  # River
        neighbors = get_neighbors(x, y, heightmap)
        if not neighbors:
            break
        next_x, next_y = min(neighbors, key=lambda pos: heightmap[pos[0], pos[1]])
        if heightmap[next_x][next_y] >= heightmap[x][y]:
            break
        x, y = next_x, next_y


def add_lakes(terrain, heightmap, config):
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if heightmap[i][j] < config["WATER_THRESHOLD"] and terrain[i][j] != 5:
                terrain[i][j] = 6  # Lake
    return terrain


def add_ponds(terrain, config):
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if terrain[i][j] in (1, 2) and random.random() < config["POND_PROBABILITY"]:
                terrain[i][j] = 7  # Pond
    return terrain


def add_caves(terrain, heightmap, config):
    for i in range(terrain.shape[0]):
        for j in range(terrain.shape[1]):
            if (
                heightmap[i][j] > config["PLAINS_THRESHOLD"]
                and random.random() < config["CAVE_PROBABILITY"]
            ):
                terrain[i][j] = 9  # Cave
    return terrain

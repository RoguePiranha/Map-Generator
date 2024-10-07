import random
from collections import deque

def place_villages(terrain, num_villages, village_radius=3):
    village_positions = []

    for _ in range(num_villages):
        retries = 0
        while retries < 100:  # Limit retries to avoid infinite loops
            x = random.randint(0, terrain.shape[0] - 1)
            y = random.randint(0, terrain.shape[1] - 1)

            # Ensure we only place villages on plains (value = 2)
            if terrain[x][y] == 2:
                village_positions.append((x, y))

                # Create an organic-shaped village with flood-fill
                village_radius = village_radius_randomizer(village_radius)
                expand_village(terrain, x, y, village_radius)
                break

            retries += 1

    return village_positions


def village_radius_randomizer(village_radius):
    """Randomize village radius."""
    return random.randint(village_radius // 2, village_radius * 2)


def expand_village(terrain, x, y, village_radius):
    """Expand the village organically using a flood-fill-like approach."""
    queue = deque([(x, y)])
    visited = set()
    visited.add((x, y))
    
    while queue and len(visited) < village_radius:
        cx, cy = queue.popleft()

        # Mark the current cell as part of the village
        if terrain[cx][cy] == 2:  # Only expand into plains (value 2)
            terrain[cx][cy] = 4  # Mark as village
        
        # Check neighboring cells to grow the village
        for nx, ny in get_neighbors(cx, cy, terrain):
            if (nx, ny) not in visited and terrain[nx][ny] == 2:
                visited.add((nx, ny))
                queue.append((nx, ny))


def get_neighbors(x, y, terrain):
    """Get valid neighboring coordinates (4 directions: up, down, left, right)."""
    neighbors = []
    if x > 0:
        neighbors.append((x - 1, y))
    if x < terrain.shape[0] - 1:
        neighbors.append((x + 1, y))
    if y > 0:
        neighbors.append((x, y - 1))
    if y < terrain.shape[1] - 1:
        neighbors.append((x, y + 1))
    return neighbors


# Function to add rivers, lakes, etc. stays the same

# Removing the cliffs functionality (not needed anymore)

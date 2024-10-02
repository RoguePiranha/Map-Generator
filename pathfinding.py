import heapq
from utils import get_neighbors

# Define movement costs for different terrains
TERRAIN_COSTS = {
    0: 100,  # Water (impassable or very expensive)
    1: 3,  # Forest (medium difficulty)
    2: 1,  # Plains (easy to travel)
    3: 100,  # Mountain (impassable or very expensive)
    4: 1,  # Village (destination)
    5: 1,  # River (can be crossed, but slightly more costly)
    6: 50,  # Lake (impassable or very expensive)
    7: 2,  # Pond (slightly higher cost)
    8: 1,  # Road (already established, easy to follow)
    9: 100,  # Caves (impassable or very expensive)
    10: 100,  # Cliffs (impassable or very expensive)
    11: 100,  # Canyons (impassable or very expensive)
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

import heapq


# Define movement costs for different terrains
TERRAIN_COSTS = {
    0: 100,  # Water (impassable or very expensive)
    1: 3,    # Forest (medium difficulty)
    2: 1,    # Plains (easy to travel)
    3: 100,  # Mountain (impassable or very expensive)
    4: 1,    # Village (destination)
    5: 1,    # River (can be crossed, but slightly more costly)
    6: 50,   # Lake (impassable or very expensive)
    7: 2,    # Pond (slightly higher cost)
    8: 1,    # Road (already established, easy to follow)
}

# A* pathfinding algorithm
def a_star(start, goal, terrain):
    """A* pathfinding algorithm."""
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
            new_cost = cost_so_far[current] + TERRAIN_COSTS[terrain[neighbor[0]][neighbor[1]]]
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

def get_neighbors(x, y, terrain):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            if 0 <= x + i < terrain.shape[0] and 0 <= y + j < terrain.shape[1]:
                neighbors.append((x + i, y + j))
    return neighbors

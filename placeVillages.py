import random


def place_villages(terrain, num_villages, village_radius=3):
    """
    Place villages on the map with a specified radius to make them larger.
    Marks a region around the village center as a village.

    :param terrain: The terrain grid.
    :param num_villages: Number of villages to place.
    :param village_radius: Radius of the village (default is 3).
    :return: List of village center coordinates.
    """
    village_positions = []

    for _ in range(num_villages):
        while True:
            x = random.randint(0, terrain.shape[0] - 1)
            y = random.randint(0, terrain.shape[1] - 1)

            # Ensure we only place villages on plains (value = 2)
            if terrain[x][y] == 2:
                village_positions.append((x, y))

                # Mark the area around the village center as part of the village
                for i in range(-village_radius, village_radius + 1):
                    for j in range(-village_radius, village_radius + 1):
                        if (
                            0 <= x + i < terrain.shape[0]
                            and 0 <= y + j < terrain.shape[1]
                        ):
                            terrain[x + i][y + j] = 4  # Mark as village
                break

    return village_positions


def village_radius(village_radius):
    """
    Set the radius of the village.

    :param village_radius: Radius of the village.
    """
    # set village radius to be a value of between the village radius and 10 above or below
    village_radius = random.randint(village_radius - 10, village_radius + 10)

    return village_radius

def create_gradient_color_map():
    from matplotlib import colors

    # Create a continuous colormap for the heightmap (elevation)
    cmap = colors.LinearSegmentedColormap.from_list(
        'terrain_map', 
        [
            (0.0, "darkblue"),   # Deep water
            (0.25, "blue"),      # Shallow water
            (0.3, "cyan"),       # Lakes and water bodies
            (0.35, "beige"),     # Beach/sand
            (0.4, "lightgreen"), # Plains
            (0.6, "green"),      # Forest
            (0.75, "darkgreen"), # Dense forest
            (0.8, "brown"),      # Mountains
            (1.0, "white")       # Mountain peaks or snow
        ]
    )
    return cmap
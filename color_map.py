def create_gradient_color_map():
    from matplotlib import colors

    # Breakpoints aligned with terrain thresholds:
    #   Water  < 0.28
    #   Forest   0.28 - 0.45
    #   Plains   0.45 - 0.72
    #   Mountain > 0.72
    cmap = colors.LinearSegmentedColormap.from_list(
        'terrain_map',
        [
            (0.0,  "#001a4d"),   # Deep ocean
            (0.15, "#003399"),   # Ocean
            (0.25, "#2266bb"),   # Shallow water
            (0.28, "#d2b98b"),   # Shoreline / beach
            (0.32, "#1a6622"),   # Dense forest
            (0.38, "#228B22"),   # Forest
            (0.45, "#90c060"),   # Forest-to-plains transition
            (0.55, "#a8d88c"),   # Light plains
            (0.65, "#c8e6a0"),   # Open plains
            (0.72, "#8B7355"),   # Foothills
            (0.80, "#6B4226"),   # Mountains
            (0.90, "#888888"),   # High rocky peaks
            (1.0,  "#f0f0f0"),  # Snow caps
        ]
    )
    return cmap
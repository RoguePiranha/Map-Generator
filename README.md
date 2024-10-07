
# 🗺️ Map Generator - Procedural Terrain and Map Creation

Welcome to the **Map Generator**, a Python-based project that procedurally generates 2D maps for game development, Dungeons & Dragons (DnD), and other creative uses. Using the power of **Perlin noise**, this project generates rich and realistic terrains complete with features like **rivers**, **lakes**, **forests**, **mountains**, **valleys**, **villages**, **roads**, **ponds**, **cliffs**, **canyons**, and **caves**.

This project is in progress and improvements are being worked on. Any suggestions are greatly appreciated!

## 🎮 Features

- **Procedural Terrain Generation**:
  - Uses **Perlin noise** to create smooth and organic terrain heightmaps.
  - Supports customizable terrain types including **mountains**, **plains**, **forests**, and **water bodies** (lakes, rivers, ponds).
  
- **Geographical Features**:
  - Adds **rivers** that flow naturally from higher elevations to lowlands.
  - **Lakes** and **ponds** are generated based on elevation thresholds, providing a diverse water system.
  - Generate **cliffs** and **canyons** based on sharp elevation changes.
  - Includes **caves** randomly placed in elevated areas like mountains and hills.

- **Villages and Roads**:
  - Place **villages** in strategic locations (plains) and create realistic **roads** connecting them using an **A* pathfinding algorithm**.

- **Fully Customizable**:
  - Configurable terrain parameters (like **octaves**, **scale**, **persistence**, and **lacunarity**) to adjust the Perlin noise and fine-tune the terrain generation.
  - Adjust the size of villages, cliffs, and canyons easily via a configuration dictionary.

- **Colorful Visualization**:
  - Visualize the generated maps using **matplotlib** with a fully customizable color scheme for each terrain feature.

## 🚀 Quick Start

### Prerequisites

To run the project, you'll need to have the following dependencies installed:

- Python 3.6+
- Required Python packages (can be installed via pip):
  
  ```bash
  pip install numpy noise matplotlib perlin-noise
  ```

### Running the Map Generator

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/map-generator.git
   cd map-generator
   ```

2. **Modify Configuration**:
   - Open the `main.py` file and adjust the configuration dictionary to tweak the map generation parameters, such as the size of the map, the number of villages, and terrain thresholds.
   
3. **Generate the Map**:
   - Run the `main.py` script to generate and visualize the map:

   ```bash
   python main.py
   ```

4. **Explore the Map**:
   - The generated map will be visualized using `matplotlib`, allowing you to see the detailed terrain, geographical features, and villages.

## 🛠️ Configuration

The map generation process is controlled via a configuration dictionary in `main.py`. Here are the key parameters you can adjust:

```python
config = {
    "WIDTH": 1920,              # Width of the map
    "HEIGHT": 1080,             # Height of the map
    "SCALE": 100.0,            # Scale factor for Perlin noise
    "OCTAVES": 6,              # Number of noise layers
    "PERSISTENCE": 0.5,        # Controls the amplitude of each octave
    "LACUNARITY": 2.0,         # Controls the frequency of each octave
    "SEED": 42,                # Seed for reproducibility

    # Terrain thresholds
    "MOUNTAIN_THRESHOLD": 0.6, # Elevation threshold for mountains
    "PLAINS_THRESHOLD": 0.4,   # Elevation threshold for plains
    "WATER_THRESHOLD": 0.3,    # Elevation threshold for water bodies

    # Feature-specific settings
    "RIVER_THRESHOLD": 0.25,   # Minimum height for river start
    "NUM_VILLAGES": 15,         # Number of villages to place
    "VILLAGE_RADIUS": 10,        # Radius for village size
    "POND_PROBABILITY": 0.01,  # Probability of placing a pond
    "CAVE_PROBABILITY": 0.02,  # Probability of placing a cave
    "CLIFF_THRESHOLD": 0.05,   # Steepness required for cliffs
}
```

## 🌍 Terrain Features

| Feature        | Description                                                                              |
|----------------|------------------------------------------------------------------------------------------|
| **Mountains**  | Elevation-based highlands. Configurable by adjusting `MOUNTAIN_THRESHOLD`.                |
| **Plains**     | Mid-elevation areas, perfect for villages. Configurable via `PLAINS_THRESHOLD`.           |
| **Forests**    | Dense areas of trees and foliage in low to mid-elevation areas.                           |
| **Rivers**     | Naturally flowing rivers generated from higher elevations to lowlands.                    |
| **Lakes**      | Larger water bodies found in low-elevation regions.                                       |
| **Ponds**      | Small, scattered water bodies placed based on probability (`POND_PROBABILITY`).           |
| **Cliffs**     | Sharp elevation changes identified as cliffs. Adjust with `CLIFF_THRESHOLD`.              |
| **Canyons**    | Low-lying, steep-sided valleys near rivers or cliffs.                                     |
| **Caves**      | Rare, deep underground structures located in mountainous regions (`CAVE_PROBABILITY`).    |
| **Villages**   | Populated areas placed on plains (`NUM_VILLAGES`). Configurable village size (`VILLAGE_RADIUS`). |
| **Roads**      | Roads connecting villages using A* pathfinding.                                           |

## 🎨 Customizing Terrain Colors

The colors for each terrain feature are fully customizable in the `terrain_colors` dictionary in `main.py`:

```python
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
```

## 🧠 How It Works

### Perlin Noise
The terrain is generated using **Perlin noise**, a gradient noise function that produces smooth, natural-looking variations in height. The noise values are used to determine the elevation of different points on the map, which are then classified into different terrain types (e.g., mountains, plains, forests, water).

### A* Pathfinding
Villages are connected by roads that are generated using the **A* pathfinding algorithm**. This algorithm ensures that roads take the shortest path between villages, avoiding impassable terrain like mountains or water.

## 💡 Future Enhancements

- **Biome Support**: Add support for different biomes (e.g., desert, tundra, jungle) based on the generated elevation and temperature.
- **Weather Effects**: Simulate weather patterns such as rain or snow and their effects on the terrain.
- **Caves Expansion**: Add tunnels and complex underground cave systems for deeper exploration.

## 👥 Contributing

Contributions are welcome! If you have ideas to improve the project or want to add new features, feel free to fork the repository and submit a pull request.

1. Fork the repository.
2. Create a new branch.
3. Submit a pull request with your changes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

🌟 **Enjoy creating procedurally generated worlds with this Map Generator!** 🌍

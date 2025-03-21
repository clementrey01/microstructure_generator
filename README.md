# Microstructure Generator

The Microstructure Generator is a tool designed to generate microstructures with random porosities. So far, it is only able to generate sphere packings.

## Project Structure

- **`main.py`**: The main entry point of the application. It parses command-line arguments and orchestrates the microstructure generation process.
- **`configuration/config.py`**: Contains the configuration parameters and methods to handle them.
- **`example_params/`**: Directory containing example JSON parameter files ready to use for different simulations.
- **`export/vtk_exporter.py`**: Handles the export of generated microstructures to VTK format for visualization.
- **`structures/`**: Contains the structure-related classes and implementations.
  - **`base_structure.py`**: Defines the abstract base class common to all structures.
  - **`sphere_packing.py`**: Implements the sphere packing structure, supporting 2D and 3D configurations with various size distributions (monodisperse, normal, lognormal, uniform, custom).
- **`visualization/visualize.py`**: Provides functions to plot and visualize the generated microstructures.
- **`utils/io.py`**: Contains utility functions for input/output operations.

## Usage

### Command Line Arguments

The project can be run from the command line with various options:

- **Basic Usage**:
  ```console
  python3 main.py
  ```

This command runs the generator with standard parameters and no visualization.

- **With Configuration File**:
  ```console
  python3 main.py --config example_params/example.json
  ```

Specify a configuration file to use custom parameters.

- **With Visualization**:
  ```console
  python3 main.py --visualize
  ```

Enable visualization of the generated microstructure.

- **Combined Options**:
  ```console
  python3 main.py -c example_params/example.json -v
  ```

Use a custom configuration file and enable visualization.


### Example JSON Parameter Files

The `example_params/` directory contains several example JSON files for different configurations:

- `2d_sphere_monodisperse.json`: 2D monodisperse sphere packing.
- `3d_sphere_monodisperse.json`: 3D monodisperse sphere packing.
- `3d_sphere_uniform_filename.json`: 3D polydisperse sphere packing with a uniform size distribution and a custom filename.
- `2d_sphere_custom.json`: 2D polydisperse sphere packing with a custom size distribution.
- `3d_sphere_lognormal.json`: 3D polydisperse sphere packing with a lognormal size distribution.
- `2d_sphere_normal.json`: 2D polydisperse sphere packing with a normal size distribution.
- `2d_sphere_uniform.json`: 2D polydisperse sphere packing with a uniform size distribution.


### Parameters

The following parameters can be configured in the JSON files:

- `structure_type`: Type of structure to generate (e.g., `sphere_packing`).
- `is_3d`: Boolean indicating whether the structure is 3D.
- `domain_size`: List specifying the size of the domain. For example, `[1.0, 1.0]` in 2D or `[1.0, 1.0, 1.0]` in 3D.
- `resolution`: List specifying the resolution of the domain (e.g., `[100, 100]` in 2D).
- `target_porosity`: Target porosity of the structure (e.g., `0.2` for 20%).
- `max_attempts`: Maximum number of attempts to place each sphere (e.g., `1000`).
- `batch_size`: Number of positions generated per loop, for parallel checking (e.g., `100`).
- `margin`: Margin between porosities, and between porosities and boundary (e.g., `5e-3`).
- `size_distribution`: Dictionary specifying the size distribution parameters. Detail in `example_params/`.
- `move_factor`: Force factor of the paxking relaxation process.
- `show_plot`: Boolean indicating whether to show the plot.
- `directory_path`: Path to the directory where output files will be saved. For example, `../../data/generated_microstructures/`.
- `filename`: Base name for the output files.


### Visualization

The `visualization/visualize.py` module provides functions to plot the generated microstructures. It supports both 2D and 3D visualizations, including wireframe representations of spheres.


### Export

The `export/vtk_exporter.py` module allows exporting the generated microstructures to VTK format, which can be visualized using tools like ParaView.

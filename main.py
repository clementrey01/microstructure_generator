#!/usr/bin/env python3

"""
========================================================================================================================
MAIN - MICROSTRUCTURE GENERATOR
========================================================================================================================
Description:
------------
This programs generates microstructures with specified geometry and topology.
"""

import argparse
import logging
from configuration.config import StructureParams
from structures.sphere_packing import SpherePacking
from export.vtk_exporter import VTKexporter
from visualization.visualizer import Visualizer
from utils.io import (
    save_data,
    save_metadata,
    load_params_from_file,
    create_unique_output_path,
)


def setup_logging(log_level="INFO"):
    """Set up logging configuration"""

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_arguments():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser(description="Microstructure Generator")
    parser.add_argument("-c", "--config", type=str, help="Path to configuration file")
    parser.add_argument(
        "-v", "--visualize", action="store_true", help="Visualize the microstructure"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main():
    """Main function to execute the microstructure generation pipeline"""

    # Parse command line arguments
    args = parse_arguments()
    setup_logging(args.log_level)

    # Load arg parameters - json parameters file and visualization
    if args.config:
        params = load_params_from_file(args.config)
        logging.info(f"Loaded parameters from {args.config}")
    else:
        # Use default parameters
        params = StructureParams()
        logging.info("Using default parameters")

    # Generate microstructure
    if params.is_3d:
        dimension = "3d"
    else:
        dimension = "2d"

    logging.info(f"Generating {dimension} {params.structure_type} microstructure...")

    if params.structure_type == "sphere_packing":
        # Initialize sphere packing
        structure = SpherePacking(domain_size=params.domain_size, is_3d=params.is_3d)

        # Generate sphere packing

        structure.generate(
            size_distribution=params.size_distribution,
            target_porosity=params.target_porosity,
            max_attempts=params.max_attempts,
            batch_size=params.batch_size,
            move_factor=params.move_factor,
            margin=params.margin,
        )

        # Create voxel grid
        grid = structure.create_voxel_grid(params.resolution)
        logging.info(f"Created voxel grid with {len(grid)} points")

        # Assign material and zone data
        material_data, zone_data = structure.assign_material_and_zone(
            grid, params.resolution
        )
        logging.info("Assigned material and zone data")

        if params.is_3d:
            dim_var = "3d"
        else:
            dim_var = "2d"

        # Create folder and filename
        if params.filename is None:
            output_filename = f"{dim_var}_{params.structure_type}_{params.size_distribution.get('type', 'monodisperse')}_{params.domain_size}_{structure.porosity:.2f}"
        else:
            output_filename = params.filename
        output_directory = create_unique_output_path(
            params.directory_path, output_filename
        )

        # Export to VTK
        VTKexporter.export_vtk_structured(
            output_filename,
            output_directory,
            material_data,
            zone_data,
            params.resolution,
        )

        # Save metadata
        formatted_porosity = f"{structure.porosity:.3f}"
        metadata = {
            "structure_type": params.structure_type,
            "is_3d": params.is_3d,
            "size_distribution": params.size_distribution,
            "domain_size": params.domain_size,
            "resolution": params.resolution,
            "target_porosity": params.target_porosity,
            "porosity": formatted_porosity,
            "num_spheres": len(structure.centers)
            if structure.centers is not None
            else 0,
            "max_attempts": params.max_attempts,
            "batch_size": params.batch_size,
            "margin": params.margin,
            "move_factor": params.move_factor,
        }

        save_metadata(metadata, output_filename, output_directory)

        # Save data
        data = {"radii": structure.radii, "centers": structure.centers}

        save_data(data, output_filename, output_directory)

        # Visualize if requested
        if args.visualize:
            logging.info("Visualizing microstructure...")
            viz_data = structure.visualize()
            Visualizer().plot_sphere_packing(
                viz_data,
                show=params.show_plot,
                directory=output_directory,
                filename=output_filename,
            )
            Visualizer().plot_size_distribution(
                structure.radii,
                bins=30,
                show=True,
                directory=output_directory,
                filename=output_filename,
            )

    else:
        logging.error(f"Unsupported structure type: {params.structure_type}")
        return

    logging.info("Microstructure generation complete")


if __name__ == "__main__":
    main()

"""
=============================================================================================================
VTK EXPORTER
=============================================================================================================
Description:
------------
This module contains the class for exporting microstructures to VTK format.
"""

import os
import logging
import struct
import numpy as np


class VTKexporter:
    """
    Class for exporting microstructures data to VTK format.
    """

    @staticmethod
    def export_vtk_structured(
        filename: str,
        directory: str,
        material_data: np.ndarray,
        zone_data: np.ndarray,
        resolution: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        """
        Export structured grid to VTK format.

        Parameters:
        -----------
        filename : str
            Name of the output file, without extension.
        directory: str
            Name of the output directory
        material_data : numpy.ndarray
            Material ID for each grid point (0=matrix, 1=pore).
        zone_data : numpy.ndarray
            Zone ID for each grid point (0=matrix, >0=specific pore ID).
        resolution : tuple[int, int] | tuple[int, int, int]
            Number of points in each dimension.

        Returns:
        --------
        None
        """

        # Export material ID data
        material_filename = os.path.join(directory, filename + "_materialID.vtk")
        VTKexporter._export_vtk_amitex(
            material_filename, material_data, "materialID", resolution
        )

        # Export zone ID data
        zone_filename = os.path.join(directory, filename + "_zoneID.vtk")
        VTKexporter._export_vtk_amitex(zone_filename, zone_data, "zoneID", resolution)

        logging.info(f"Successfully exported VTK files to {filename}.")

    @staticmethod
    def _export_vtk_amitex(
        filename: str,
        field_data: np.ndarray,
        field_name: str,
        resolution: tuple[int, int] | tuple[int, int, int],
        origin=(0.0, 0.0, 0.0),
        spacing=(1.0, 1.0, 1.0),
        data_type="unsigned_short",
    ) -> None:
        """
        Export VTK file for AMITEX_FFTP
                Parameters:
        -----------
        filename : str
            Name of the output file, with extension.
        field_data : numpy.ndarray
            Field data to be exported.
        field_name : str
            Name of the field.
        resolution : tuple[int, int] | tuple[int, int, int]
            Number of points in each dimension.
        origin : tuple[float, float, float]
            Origin of the structure. By default, (0.0, 0.0, 0.0).
        spacing: tuple[float, float, float]
            Spacing between voxels. By default, (1.0, 1.0, 1.0)
        data_type: str
            Type of the encoded data. By default, unsigned_short.

        Returns:
        --------
        None
        """

        # Validate that material or zone IDs are properly numbered (consecutive with no gaps)
        if field_name.lower() == "materialid":
            # Get unique material IDs, excluding 0 if present
            unique_ids = np.unique(field_data)
            if 0 in unique_ids:
                unique_ids = unique_ids[unique_ids != 0]

            # Check if IDs are consecutive
            expected_ids = np.arange(1, len(unique_ids) + 1)
            if not np.array_equal(unique_ids, expected_ids):
                logging.warning(
                    f"Material IDs should be consecutive from 1 to {len(unique_ids)}"
                )
                logging.warning(f"Found IDs: {unique_ids}")

                # Auto-fix material IDs
                old_to_new = {old: new for old, new in zip(unique_ids, expected_ids)}
                for i in range(field_data.size):
                    if field_data.flat[i] != 0:  # Keep 0 as is
                        field_data.flat[i] = old_to_new[field_data.flat[i]]

                logging.info("Material IDs have been renumbered to be consecutive")

        # Determine if the field is 2D or 3D
        is_3d = len(resolution) == 3

        if len(resolution) == 3:
            nx, ny, nz = resolution
        else:
            nx, ny = resolution
            nz = 1

        # Calculate number of cells
        cell_count = nx * ny * nz

        # For STRUCTURED_POINTS, the dimensions in the header must be nx+1, ny+1, nz+1
        # because DIMENSIONS in the header refers to number of points, not cells
        vtk_dims = (nx + 1, ny + 1, nz + 1) if is_3d else (nx + 1, ny + 1, 1 + 1)

        # Ensure field_data has correct shape
        if field_data.size != cell_count:
            raise ValueError(
                f"Field data size ({field_data.size}) doesn't match cell count ({cell_count})"
            )

        # Flatten the field data in the correct order (Fortran order for VTK)
        if field_data.ndim > 1:
            field_data_flat = field_data.ravel(order="F")
        else:
            field_data_flat = field_data

        # Type mapping for Python struct
        type_map = {
            "char": "b",
            "short": "h",
            "int": "i",
            "long": "l",
            "unsigned_char": "B",
            "unsigned_short": "H",
            "unsigned_int": "I",
            "unsigned_long": "L",
            "float": "f",
            "double": "d",
        }

        if data_type not in type_map:
            raise ValueError(f"Unsupported data type: {data_type}")

        struct_format = ">" + type_map[data_type]  # '>' for big-endian

        # Write binary VTK file
        with open(filename, "wb") as f:
            # Write header (ASCII part) - ensuring exact format required by AMITEX_FFTP
            header = f"""# vtk DataFile Version 4.5
{field_name}
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS    {vtk_dims[0]}   {vtk_dims[1]}   {vtk_dims[2]}
ORIGIN    {origin[0]:.3f}   {origin[1]:.3f}   {origin[2]:.3f}
SPACING    {spacing[0]:.6f}    {spacing[1]:.6f}   {spacing[2]:.6f}
CELL_DATA   {cell_count}
SCALARS {field_name} {data_type}
LOOKUP_TABLE default
"""
            f.write(header.encode("ascii"))

            # Write binary data in big-endian format
            for value in field_data_flat:
                f.write(struct.pack(struct_format, int(value)))

        logging.info(f"Successfully exported AMITEX-compatible VTK file to {filename}.")

    @staticmethod
    def _export_vtk_field(
        filename: str,
        grid: np.ndarray,
        field_data: np.ndarray,
        field_name: str,
        resolution: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        """
        Export a single scalar field to VTK format.

        Parameters:
        -----------
        filename : str
            Name of the output file, with extension.
        grid : numpy.ndarray
            Array of points in the domain.
        field_data : numpy.ndarray
            Field data to be exported.
        field_name : str
            Name of the field.
        resolution : tuple[int, int] | tuple[int, int, int]
            Number of points in each dimension.

        Returns:
        --------
        None
        """
        # Determine if the field is 2D or 3D
        is_3d = len(resolution) == 3

        if len(resolution) == 3:
            nx, ny, nz = resolution
        else:
            nx, ny = resolution
            nz = 1

        # Calculate spacing
        if is_3d:
            # Find extents
            x_min, y_min, z_min = grid.min(axis=0)
            x_max, y_max, z_max = grid.max(axis=0)

            # Calculate grid spacing
            dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1
            dy = (y_max - y_min) / (ny - 1) if ny > 1 else 1
            dz = (z_max - z_min) / (nz - 1) if nz > 1 else 1
        else:
            # Find extents
            x_min, y_min = grid.min(axis=0)
            x_max, y_max = grid.max(axis=0)

            # Calculate grid spacing
            dx = (x_max - x_min) / (nx - 1) if nx > 1 else 1
            dy = (y_max - y_min) / (ny - 1) if ny > 1 else 1
            dz = 1  # Default z spacing for 2D
            z_min = 0  # Default z position for 2D

        # Total number of points
        npoin = nx * ny * nz

        # Open file
        with open(filename, "w") as f:
            # Write header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"VTK structured grid for {field_name}\n")
            f.write("ASCII\n")

            # Use STRUCTURED_GRID format like in the working example
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
            f.write(f"POINTS {npoin} float\n")

            # Write grid points
            if is_3d:
                # Recreate the grid points explicitly
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            x = x_min + i * dx
                            y = y_min + j * dy
                            z = z_min + k * dz
                            f.write(f"{x} {y} {z}\n")
            else:
                # Recreate the grid points explicitly (2D case)
                for i in range(nx):
                    for j in range(ny):
                        x = x_min + i * dx
                        y = y_min + j * dy
                        z = z_min  # Fixed z for 2D
                        f.write(f"{x} {y} {z}\n")

            # Write field data
            f.write(f"POINT_DATA {npoin}\n")
            f.write(f"SCALARS {field_name} int 1\n")
            f.write("LOOKUP_TABLE default\n")

            # Write data values in the correct order
            if is_3d:
                # Reshape field_data if needed
                if field_data.ndim == 1:
                    # Assuming field_data is flattened in the correct order
                    for value in field_data:
                        f.write(f"{int(value)}\n")
                else:
                    # Assuming field_data is 3D array
                    for i in range(nx):
                        for j in range(ny):
                            for k in range(nz):
                                try:
                                    value = field_data[i, j, k]
                                except IndexError:
                                    # If the data is not in the expected shape, try flattened access
                                    idx = i * ny * nz + j * nz + k
                                    if idx < len(field_data):
                                        value = field_data[idx]
                                    else:
                                        value = 0  # Default value
                                f.write(f"{int(value)}\n")
            else:
                # Handle 2D data
                if field_data.ndim == 1:
                    # Assuming field_data is flattened in the correct order
                    for value in field_data:
                        f.write(f"{int(value)}\n")
                else:
                    # Assuming field_data is 2D array
                    for i in range(nx):
                        for j in range(ny):
                            try:
                                value = field_data[i, j]
                            except IndexError:
                                # If the data is not in the expected shape, try flattened access
                                idx = i * ny + j
                                if idx < len(field_data):
                                    value = field_data[idx]
                                else:
                                    value = 0  # Default value
                            f.write(f"{int(value)}\n")

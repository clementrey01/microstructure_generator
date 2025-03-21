"""
========================================================================================================================
PARAMETERS
========================================================================================================================
Description:
------------
This module contains the functions to read and export the parameters
"""

import json


class StructureParams:
    """
    Class to store and manage parameters for microstructure generation.
    """

    def __init__(self) -> None:
        """Initialize with default parameters"""
        # General parameters
        self.structure_type = "sphere_packing"  # 'sphere_packing', etc.
        self.is_3d = True
        self.domain_size = [1.0, 1.0, 1.0]  # [x, y, z] or [x, y] for 2D
        self.resolution = [100, 100, 100]  # Grid resolution
        self.target_porosity = 0.4  # Target porosity
        self.max_attempts = 1000  # Maximum placement attempts
        self.batch_size = 100  # Batch per placement
        self.margin = 5e-3  # Margin between porosities, and boundary
        # Sphere packing parameters
        if self.structure_type == "sphere_packing":
            # Size distribution for packing
            self.size_distribution = {
                "type": "monodisperse",  # 'monodisperse', 'lognormal', 'uniform', or 'custom'
                "radius": 0.1,  # Radius
            }

        # Move factor for packing relaxation
        self.move_factor = 0.05

        # Visualization parameters
        self.show_plot = False  # Show plot interactively

        # Storing the data
        self.directory_path = "."  # Directory path to save the data
        self.filename = None  # Filename to save the data

    def to_dict(self) -> dict:
        """Convert parameters to dictionary"""
        return {
            "structure_type": self.structure_type,
            "is_3d": self.is_3d,
            "domain_size": self.domain_size,
            "resolution": self.resolution,
            "target_porosity": self.target_porosity,
            "max_attempts": self.max_attempts,
            "batch_size": self.batch_size,
            "margin": self.margin,
            "size_distribution": self.size_distribution,
            "move_factor": self.move_factor,
            "show_plot": self.show_plot,
            "directory_path": self.directory_path,
            "filename": self.filename,
        }

    def from_dict(self, params_dict: dict) -> object:
        """Load parameters from dictionary"""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def save(self, filename: str) -> None:
        """Save parameters to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filename: str) -> object:
        """Load parameters from JSON file"""
        with open(filename, "r") as f:
            params_dict = json.load(f)

        params = cls()
        return params.from_dict(params_dict)

    def validate(self):
        """Validate parameters"""
        # Check domain size
        if self.is_3d and len(self.domain_size) != 3:
            return False, "3D domain requires 3 dimensions"
        elif not self.is_3d and len(self.domain_size) < 2:
            return False, "2D domain requires at least 2 dimensions"

        # Check resolution
        if self.is_3d and len(self.resolution) != 3:
            return False, "3D domain requires 3D resolution"
        elif not self.is_3d and len(self.resolution) < 2:
            return False, "2D domain requires at least 2D resolution"

        # Check porosity
        if not 0 < self.target_porosity < 1:
            return False, "Porosity must be between 0 and 1"

        # Check max attempts
        if self.max_attempts < 1:
            return False, "Max attempt cannot be smaller than 1"

        # Check batch size
        if self.batch_size < 1:
            return False, "Batch size cannot be smaller than 1"
        elif self.batch_size > self.max_attempts:
            return False, "Batch size cannot be greater than max attempts"

        # Specific checks for sphere packing
        if self.structure_type == "sphere_packing":
            if self.size_distribution["type"] == "monodisperse":
                if self.size_distribution["radius"] <= 0:
                    return False, "Radius must be positive"
                if self.size_distribution["radius"] >= min(self.domain_size) / 2:
                    return False, "Radius can't be bigger than half the domain size"

            elif self.size_distribution["type"] == "lognormal":
                if self.size_distribution["mean"] <= 0:
                    return False, "Mean radius must be positive"
                if self.size_distribution["sigma"] <= 0:
                    return False, "Sigma must be positive"
                if self.size_distribution["max_radius"] >= min(self.domain_size) / 2:
                    return False, "Max radius can't be bigger than half the domain size"

            elif self.size_distribution["type"] == "uniform":
                if self.size_distribution["min_radius"] <= 0:
                    return False, "Minimum radius must be positive"
                if (
                    self.size_distribution["max_radius"]
                    <= self.size_distribution["min_radius"]
                ):
                    return False, "Maximum radius must be greater than minimum radius"
                if self.size_distribution["max_radius"] >= min(self.domain_size) / 2:
                    return False, "Max radius can't be bigger than half the domain size"

            elif self.size_distribution["type"] == "normal":
                if self.size_distribution["mean"] <= 0:
                    return False, "Mean radius must be positive"
                if self.size_distribution["sigma"] <= 0:
                    return False, "Sigma must be positive"
                if self.size_distribution["max_radius"] >= min(self.domain_size) / 2:
                    return False, "Max radius can't be bigger than half the domain size"

            elif self.size_distribution["type"] == "custom":
                if (
                    "radii" not in self.size_distribution
                    or not self.size_distribution["radii"]
                ):
                    return False, "Custom distribution must specify radii"
                if any(r <= 0 for r in self.size_distribution["radii"]):
                    return False, "All radii must be positive"
                if max(self.size_distribution["radii"]) >= min(self.domain_size) / 2:
                    return False, "Max radius can't be bigger than half the domain size"

            else:
                return (
                    False,
                    f"Unknown size distribution type: {self.size_distribution['type']}",
                )

        else:
            return False, f"Unsupported structure type: {self.structure_type}"

        return True, "Parameters are valid"

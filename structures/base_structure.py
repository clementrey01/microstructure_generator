"""
=============================================================================================================
STRUCTURES GENERATOR
=============================================================================================================
Description:
------------
This module contains the abstract base class for all microstructure generators. All microstructure generators
should inherit from this class.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")  # Generic type for abstract methods


class BaseStructure(ABC, Generic[T]):
    """
    Abstract base class for all microstructure generators.

    This class defines the structure and all functionalities common for
    all microstructure generators. All microstructure generators should
    inherit from this class.
    """

    def __init__(
        self,
        domain_size: tuple[float, float] | tuple[float, float, float],
        is_3d: bool,
    ) -> None:
        """
        Initialize the microstructure generator.

        Parameters:
        -----------
        domain_size : tuple[float, float] | tuple[float, float, float]
            Size of the domain in the x, y, and optionally z directions.

        is_3d : bool
            Flag to indicate if the microstructure is 3D.
        """

        # Initialize domain size
        self.is_3d = is_3d
        if is_3d:
            self.domain_size = domain_size
        else:
            self.domain_size = domain_size[:2]

    @abstractmethod
    def generate(self, **kwargs) -> T:
        """
        Generate the microstructure.

        This method should be implemented by all subclasses.
        Necessary parameters can be found in README and in example parameter files.
        """
        pass

    def create_voxel_grid(
        self,
        resolution: tuple[int, int] | tuple[int, int, int],
    ) -> np.ndarray:
        """
        Create a regular grid of points within the domain.

        Parameters:
        -----------
        resolution : tuple[int, int] | tuple[int, int, int]
            Number of points in each dimension.

        Returns:
        --------
        grid: numpy.ndarray
            Array of points in the domain.
        """

        # Create a 2D or 3D grid
        if self.is_3d:
            if len(self.domain_size) != 3:
                raise ValueError("domain_size must have three elements for 3D.")
            if len(resolution) != 3:
                raise ValueError("resolution must have three elements for 3D.")

            x = np.linspace(0, self.domain_size[0], resolution[0])
            y = np.linspace(0, self.domain_size[1], resolution[1])
            z = np.linspace(0, self.domain_size[2], resolution[2])
            grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        else:
            if len(self.domain_size) != 2:
                raise ValueError("domain_size must have two elements for 2D.")
            if len(resolution) != 2:
                raise ValueError("resolution must have two elements for 2D.")

            x = np.linspace(0, self.domain_size[0], resolution[0])
            y = np.linspace(0, self.domain_size[1], resolution[1])
            grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

        return grid

    @abstractmethod
    def assign_material_and_zone(
        self,
        grid: np.ndarray,
        resolution: tuple[int, int] | tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign material and zone information to each point in the grid.

        Parameters:
        -----------
        grid : numpy.ndarray
            Array of points in the domain.

        resolution : tuple[int, int] | tuple[int, int, int]
            Number of points in each dimension.

        Returns:
        --------
        material_data: numpy.ndarray
            Material ID for each grid point (1=matrix, 2=pore).

        zone_data: numpy.ndarray
            Zone ID for each grid point (1=matrix, 2=pore).
        """
        pass

    @abstractmethod
    def visualize(self) -> T:
        """
        Visualize the generated microstructure.

        This method must be implemented by all subclasses.
        """
        pass

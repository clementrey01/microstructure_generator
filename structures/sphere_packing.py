"""
========================================================================================================================
SPHERE PACKING GENERATOR
========================================================================================================================
Description:
------------
This module contains the SpherePacking class, which generates random sphere packings with various size distributions.
"""

import numpy as np
import logging
import random
from scipy.stats import lognorm
from scipy.spatial import KDTree
import multiprocessing
from typing import cast, TypeVar
from concurrent.futures import ThreadPoolExecutor
from .base_structure import BaseStructure

KDTreeType = TypeVar("KDTreeType", bound=KDTree)


class SpherePacking(BaseStructure):
    """
    Sphere packing for microstructure generator.

    Generates random sphere packings with various size distributions.
    """

    def __init__(
        self,
        domain_size: tuple[float, float] | tuple[float, float, float],
        is_3d: bool,
    ) -> None:
        """
        Initialize the sphere packing generator.

        Parameters:
        -----------
        domain_size : tuple[float, float] | tuple[float, float, float]
            Size of the domain in the x, y, and optionally z directions.
        is_3d : bool
            Flag to indicate if the microstructure is 3D.
        """
        super().__init__(domain_size, is_3d)

    def generate(self, **kwargs) -> bool:
        """
        Generate sphere packing with specified parameters.
        Necessary parameters can be found in README and in example parameter files.

        Parameters:
        -----------
        **kwargs: dict
            Additional parameters for the sphere packing.

        Returns:
        --------
        bool
            Success of the generation process.
        """

        if "size_distribution" not in kwargs:
            raise ValueError("Size distribution must be specified for sphere packing.")

        # Get parameters
        size_distribution = kwargs["size_distribution"]
        target_porosity = kwargs["target_porosity"]
        max_attempts = kwargs["max_attempts"]
        batch_size = kwargs["batch_size"]
        move_factor = kwargs["move_factor"]
        margin = kwargs["margin"]

        self.generate_packing(
            size_distribution,
            target_porosity,
            max_attempts,
            batch_size,
            move_factor,
            margin,
        )
        return True

    def generate_packing(
        self,
        size_distribution: dict,
        target_porosity: float,
        max_attempts: int,
        batch_size: int,
        move_factor: float,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a sphere packing with significant performance optimizations.

        Parameters:
        -----------
        size_distribution: dict
            Dictionary with the parameters of the size distribution:
            - 'type': 'monodisperse' or 'lognormal' or 'normal' or 'uniform' or 'custom'
            - Additional parameters depending on type

        target_porosity: float
            Target porosity (void fraction) of the sphere packing.

        max_attempts: int
            Maximum number of attempts before giving up.

        batch_size: int
            Number of random guesses per loop

        move_factor: float
            Force coefficient for relaxation packing

        margin: float
            Margin between spheres, other spheres and boundary

        Returns:
        --------
        centers : numpy.ndarray
            Array of sphere centers.
        radii : numpy.ndarray
            Array of sphere radii.
        """
        # Calculate target volume
        dim = 3 if self.is_3d else 2
        domain_volume = np.prod(self.domain_size[:dim])
        target_sphere_volume = float(target_porosity * domain_volume)
        total_sphere_volume = 0

        # Initialize arrays and spatial data structure
        centers: list[list[float]] = []
        radii: list[float] = []

        # Generate candidate radii based on distribution type
        candidate_radii = self._generate_radius_distribution(
            size_distribution, target_sphere_volume
        )

        # Find max possible radius for overlap check
        dist_type = size_distribution["type"]

        if dist_type == "monodisperse":
            max_radius = size_distribution["radius"]
        elif dist_type == "custom":
            max_radius = np.max(size_distribution["radii"])
        else:
            max_radius = size_distribution["max_radius"]

        # Shuffle radii for more diversity. Useless for monodisperse but
        # it's too fast to care about it
        np.random.shuffle(candidate_radii)

        # Initialize spatial data structure for quick overlap checks
        spatial_tree = None
        max_concurrent_checks = multiprocessing.cpu_count()

        failed_attempt = False  # Relax at every step after the first failed attempt

        for radius in candidate_radii:
            # Calculate sphere volume
            sphere_volume = (
                (4 / 3) * np.pi * radius**3 if self.is_3d else np.pi * radius**2
            )

            # Check if adding this sphere would exceed target volume
            if total_sphere_volume + sphere_volume > target_sphere_volume:
                if total_sphere_volume > 0.98 * target_sphere_volume:  # Close enough
                    break
                continue

            # Apply relaxation with negative margin to improve packing if failed attempt
            if failed_attempt:
                centers, radii = self._relax_packing(
                    centers, radii, 2 * max_radius, -margin, dim, move_factor
                )

                # Rebuild spatial tree after relaxation
                if len(centers) > 0:
                    centers_array = np.array(centers)
                    spatial_tree = KDTree(centers_array)
                    centers = cast(list[list[float]], centers_array.tolist())
                else:
                    spatial_tree = None

            # Try to place the sphere using optimized approach
            placed = False
            attempts_left = max_attempts

            while not placed and attempts_left > 0:
                # Generate batches of positions for parallel checking
                batch_attempts = min(batch_size, attempts_left)
                attempts_left -= batch_attempts

                # Generate random centers for this batch
                batch_centers = np.array(
                    [
                        [
                            random.uniform(
                                radius + margin, self.domain_size[i] - (radius + margin)
                            )
                            for i in range(dim)
                        ]
                        for _ in range(batch_attempts)
                    ]
                )

                # If we have existing spheres, use spatial tree for efficient overlap detection
                if len(centers) > 0 and spatial_tree is None:
                    # Build tree only once we have enough spheres
                    centers_array = np.array(centers)
                    spatial_tree = KDTree(centers_array)
                    centers = cast(list[list[float]], centers_array.tolist())

                # Check for overlaps in parallel
                if spatial_tree is not None:
                    # Use batch query to find any potential overlaps
                    with ThreadPoolExecutor(
                        max_workers=max_concurrent_checks
                    ) as executor:
                        overlap_results = list(
                            executor.map(
                                lambda pos: self._check_overlap(
                                    pos,
                                    radius,
                                    spatial_tree,  # type : ignore
                                    centers,
                                    radii,
                                    max_radius,
                                    margin,
                                ),
                                batch_centers,
                            )
                        )

                    # Find first non-overlapping position
                    for i, (overlaps, position) in enumerate(
                        zip(overlap_results, batch_centers)
                    ):
                        if not overlaps:
                            # Add the new position
                            centers.append(position)
                            radii.append(radius)
                            # Update the spatial tree with the new point
                            spatial_tree = KDTree(centers)

                            total_sphere_volume += sphere_volume
                            placed = True
                            break
                else:
                    # For the first few spheres, we don't need a spatial tree
                    for position in batch_centers:
                        overlap = False

                        for i, center in enumerate(centers):
                            dist = np.linalg.norm(position - center)
                            if dist < radii[i] + radius + margin:
                                overlap = True
                                break

                        if not overlap:
                            # Add the new position
                            centers.append(position)
                            radii.append(radius)
                            total_sphere_volume += sphere_volume
                            placed = True
                            break

            # If we couldn't place this radius, continue to the next one
            if not placed:
                failed_attempt = True
                continue

        # Apply one final relaxation
        centers, radii = self._relax_packing(
            centers, radii, max_radius, margin, dim, move_factor
        )

        # Update class attributes
        self.centers = np.array(centers)
        self.radii = np.array(radii)
        self.porosity = total_sphere_volume / domain_volume

        # Log the results
        logging.info(
            f"Generated {len(centers)} spheres with porosity {self.porosity:.3f}."
        )
        logging.info(f"Target porosity was {target_porosity:.3f}")

        return self.centers, self.radii

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

        if self.centers is None or self.radii is None:
            raise ValueError("Sphere packing not generated yet.")

        # Initialize arrays
        material_data_ = np.ones(len(grid), dtype=int)  # Matrix by default
        zone_data_ = np.ones(len(grid), dtype=int)  # Matrix by default

        # Find grid points inside each sphere
        for i, center in enumerate(self.centers):
            # Find grid points inside the sphere
            dist = np.linalg.norm(grid - center, axis=1)
            inside = dist < self.radii[i]

            # Update material and zone data
            material_data_[inside] = 2  # Pore = 1
            zone_data_[inside] = 1  # Unique ID per sphere

        # Reshape to match the grid resolution
        reshaped_material = material_data_.reshape(
            resolution, order="F"
        )  # type : ignore
        reshaped_zone = zone_data_.reshape(resolution, order="F")  # type : ignore

        material_data = reshaped_material
        zone_data = reshaped_zone

        logging.info(
            f"Material data shape: {material_data.shape}, unique values: {np.unique(material_data)}"
        )
        logging.info(
            f"Zone data shape: {zone_data.shape}, unique values: {np.unique(zone_data)}"
        )

        return material_data, zone_data

    def _generate_radius_distribution(
        self,
        size_distribution: dict,
        target_sphere_volume: float,
    ) -> np.ndarray:
        """
        Generate the radius distribution based on the specified parameters.

        Parameters:
        -----------
        size_distribution: dict
            Dictionary with the parameters of the size distribution.

        target_sphere_volume: float
            Target volume for spheres.

        Returns:
        --------
        numpy.ndarray
            Array of candidate radii.
        """

        candidate_radii: np.ndarray
        dist_type = size_distribution["type"]

        if dist_type == "monodisperse":
            radius = size_distribution["radius"]
            # Estimate number needed with buffer
            num_estimate = (
                int(target_sphere_volume / ((4 / 3) * np.pi * radius**3) * 2)
                if self.is_3d
                else int(target_sphere_volume / (np.pi * radius**2) * 2)
            )
            candidate_radii = np.full(num_estimate, radius)

        elif dist_type == "lognormal":
            mean = size_distribution["mean"]
            sigma = size_distribution["sigma"]
            min_r = size_distribution["min_radius"]
            max_r = size_distribution["max_radius"]

            # Estimate number with buffer
            num_estimate = (
                int(target_sphere_volume / ((4 / 3) * np.pi * mean**3) * 2)
                if self.is_3d
                else int(target_sphere_volume / (np.pi * mean**2) * 2)
            )

            # Generate from lognormal distribution
            sig = sigma
            scale = np.exp(np.log(mean) - sig**2)  # Scale parameter to get desired mean
            candidate_radii = np.atleast_1d(
                lognorm.rvs(sig, scale=scale, size=num_estimate)
            ).astype(float)

            # Filter radii within bounds
            candidate_radii = candidate_radii[
                (candidate_radii >= min_r) & (candidate_radii <= max_r)
            ]

        elif dist_type == "normal":
            mean = size_distribution["mean"]
            sigma = size_distribution["sigma"]
            min_r = size_distribution["min_radius"]
            max_r = size_distribution["max_radius"]

            # Estimate number with buffer
            num_estimate = (
                int(target_sphere_volume / ((4 / 3) * np.pi * mean**3) * 2)
                if self.is_3d
                else int(target_sphere_volume / (np.pi * mean**2) * 2)
            )

            # Generate from normal distribution
            candidate_radii = np.random.normal(mean, sigma, num_estimate)

            # Filter radii within bounds
            candidate_radii = candidate_radii[
                (candidate_radii >= min_r) & (candidate_radii <= max_r)
            ]

        elif dist_type == "uniform":
            min_r = size_distribution["min_radius"]
            max_r = size_distribution["max_radius"]

            # Estimate number with buffer
            num_estimate = (
                int(target_sphere_volume / ((4 / 3) * np.pi * min_r**3) * 2)
                if self.is_3d
                else int(target_sphere_volume / (np.pi * min_r**2) * 2)
            )

            # Generate from uniform distribution
            candidate_radii = np.random.uniform(min_r, max_r, num_estimate)

        elif dist_type == "custom":
            # Custom radius values and their probabilities
            custom_radii = size_distribution["radii"]
            custom_probs = size_distribution["probabilities"]

            # Estimate number with buffer
            num_estimate = (
                int(
                    target_sphere_volume
                    / ((4 / 3) * np.pi * min(custom_radii) ** 3)
                    * 2
                )
                if self.is_3d
                else int(target_sphere_volume / (np.pi * min(custom_radii) ** 2) * 2)
            )

            # Generate from custom distribution
            candidate_radii = np.random.choice(
                custom_radii, size=num_estimate, p=custom_probs
            )

        else:
            raise ValueError(
                f"Unknown distribution type: {dist_type}. Use 'monodisperse', 'lognormal', 'normal', 'uniform', or 'custom'."
            )

        return candidate_radii

    def _check_overlap(
        self,
        position: np.ndarray,
        radius: float,
        spatial_tree: KDTree,
        centers: list[list[float]],
        radii: list[float],
        max_radius: float,
        margin: float,
    ) -> bool:
        """
        Check if a sphere overlaps with existing spheres using spatial tree.

        Parameters:
        -----------
        position: numpy.ndarray
            Center position of the sphere to check.

        radius: float
            Radius of the sphere to check.

        spatial_tree: scipy.spatial.KDTree
            KD-tree of existing sphere centers.

        centers: list[list[float]]
            List of existing center positions.

        radii: list[float]
            List of existing sphere radii.

        max_radius: float
            Maximum radius of the distribution.

        margin: float
            Extra margin to prevent touching.

        Returns:
        --------
        bool
            True if overlap exists, False otherwise.
        """

        # Query the tree for potential overlaps - only check spheres within possible overlap distance
        query_radius = radius + max_radius + margin

        # Find all points within query radius
        indices = spatial_tree.query_ball_point(position, query_radius)

        # Check each potential overlap
        for idx in indices:
            dist = np.linalg.norm(position - centers[idx])
            if dist < radius + radii[idx] + margin:
                return True

        return False

    def _relax_packing(
        self,
        centers: list[list[float]],
        radii: list[float],
        kd_radius: float,
        margin: float,
        dim: int,
        move_factor: float,
        iterations=3,
    ) -> tuple[list[list[float]], list[float]]:
        """
        Apply relaxation to improve sphere packing.

        Parameters:
        -----------
        centers: list[list[float]]
            Array of sphere centers.

        radii: list[float]
            Array of sphere radii.

        kd_radius: float
            Radius of the KDTree ball.

        margin: float
            Safety margin.

        dim: int
            Dimension of the problem.

        move_factor: float
            'Rigidity' of the spring, how much to move spheres.

        iterations: int
            Number of relaxation iterations. 3 by default.

        Returns:
        --------
        Updated centers and radii.
        """

        centers_array = np.array(centers)
        radii_array = np.array(radii)

        for _ in range(iterations):
            tree = KDTree(centers_array)

            # For each sphere, find overlaps and calculate repulsion forces
            new_centers = centers_array.copy()

            for i, (center, radius) in enumerate(zip(centers_array, radii_array)):
                # Get all potential overlaps
                query_radius = radius + kd_radius + margin
                indices = tree.query_ball_point(center, query_radius)

                # Remove self from indices
                indices = [idx for idx in indices if idx != i]

                if not indices:
                    continue

                # Calculate repulsion forces
                forces = np.zeros(dim)
                for idx in indices:
                    other_center = centers_array[idx]
                    other_radius = radii_array[idx]

                    # Vector from other sphere to this one
                    direction = center - other_center
                    distance = np.linalg.norm(direction)

                    # Check for overlap
                    min_distance = radius + other_radius + margin
                    if distance < min_distance:
                        # Normalize direction and calculate force
                        if distance > 0:  # Avoid division by zero
                            direction = direction / distance
                        else:
                            # If centers are identical, move in random direction
                            direction = np.random.uniform(-1, 1, dim)
                            direction = direction / np.linalg.norm(direction)

                        # Force proportional to overlap
                        force = direction * (min_distance - distance) * move_factor
                        forces += force

                # Apply forces to move sphere
                new_position = center + forces

                # Keep within bounds
                for j in range(dim):
                    new_position[j] = max(
                        radius + margin,
                        min(self.domain_size[j] - (radius + margin), new_position[j]),
                    )

                new_centers[i] = new_position

            # Update centers
            centers_array = new_centers

            centers = list(centers_array)
            radii = list(radii_array)

        return centers, radii

    def visualize(self) -> dict:
        """
        Visualize the sphere packing.

        Parameters:
        -----------
        show_domain : bool
            Whether to show the domain boundaries
            
        figsize : tuple
            Figure size (width, height) in inches

        Returns:
        --------
        dict: dict
            Necessary data for visualization
        """
        # This is a placeholder that will be implemented in visualization/visualizer.py
        # We'll just return the necessary data for visualization
        return {
            "centers": self.centers,
            "radii": self.radii,
            "domain_size": self.domain_size,
            "is_3d": self.is_3d,
            "porosity": self.porosity,
        }

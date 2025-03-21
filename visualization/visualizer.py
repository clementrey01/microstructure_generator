import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


class Visualizer:
    """
    Class for visualizing microstructures.
    """

    def plot_sphere_packing(
        self,
        data: dict,
        show: bool,
        directory: str,
        filename: str,
        figsize=(10, 8),
    ) -> plt.Figure:
        """
        Plot sphere packing visualization.

        Parameters:
        -----------
        data : dict
            Dictionary containing centers, radii, domain_size, is_3d, porosity

        show : bool
            Whether to show the plot.

        directory: str
            Output directory

        filename : str
            Filename of the figure (if None, figure is not saved).

        figsize : tuple
            Figure size (width, height). By default it is (10,8)

        Returns:
        --------
        fig : matplotlib figure
            The figure object
        """

        centers = data["centers"]
        radii = data["radii"]
        domain_size = data["domain_size"]
        is_3d = data["is_3d"]
        porosity = data["porosity"]

        # Make filename path
        if filename is not None:
            viz_filename = os.path.join(directory, filename + "_sphere_packing.png")
        else:
            viz_filename = None

        # Choose which function to use
        if is_3d:
            return self._plot_sphere_packing_3d(
                centers, radii, domain_size, porosity, show, viz_filename, figsize
            )
        else:
            return self._plot_sphere_packing_2d(
                centers, radii, domain_size, porosity, show, viz_filename, figsize
            )

    def _plot_sphere_packing_3d(
        self,
        centers: list[list[float]],
        radii: list[float],
        domain_size: tuple[float, float, float],
        porosity: float,
        show: bool,
        viz_filename: str,
        figsize=(10, 8),
    ) -> plt.Figure:
        """
        Plot 3D sphere packing visualization.

        Parameters:
        -----------
        centers: list
            List of all sphere centers

        radii: list
            List of all sphere radii

        domain_size: list
            List of all the dimensions

        Porosity: float
            Porosity of the structure. By default None

        show : bool
            Whether to show the plot. By default False

        save_path : str
            Path to save the figure (if None, figure is not saved). By default None

        figsize : tuple
            Figure size (width, height). By default (10,8)

        Returns:
        --------
        fig : matplotlib figure
            The figure object
        """

        # Initialize the figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        # Define a colormap of spheres based on radii
        radius_range = 0
        cmap = mpl.colormaps["viridis"]  # Standard colormap
        norm = None
        if len(radii) > 0:
            min_radius = np.min(radii)
            max_radius = np.max(radii)
            radius_range = max_radius - min_radius

            # Normalize radius values for coloring
            if radius_range > 0:
                norm = Normalize(vmin=min_radius, vmax=max_radius)

        # Plot spheres as wireframes
        for center, radius in zip(centers, radii):
            u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]  # type: ignore
            x = center[0] + radius * np.cos(u) * np.sin(v)
            y = center[1] + radius * np.sin(u) * np.sin(v)
            z = center[2] + radius * np.cos(v)

            # Determine color based on radius
            color = (
                cmap(norm(radius)) if (radius_range > 0 and norm is not None) else "b"
            )

            ax.plot_wireframe(x, y, z, color=color, alpha=0.3)  # type: ignore

        # Set axis limits - convert lists to tuples
        ax.set_xlim((0, domain_size[0]))
        ax.set_ylim((0, domain_size[1]))
        ax.set_zlim((0, domain_size[2]))  # type: ignore

        # Add axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore

        # Add title with statistics
        title = f"3D Sphere Packing with {len(centers)} spheres"
        if porosity is not None:
            title += f", Porosity of {100 * (porosity):.4f}%"
        ax.set_title(title)

        # Add colorbar for radius
        if len(radii) > 0:
            min_radius = np.min(radii)
            max_radius = np.max(radii)
            radius_range = max_radius - min_radius

            if radius_range > 0:
                cmap = mpl.colormaps["viridis"]
                norm = Normalize(vmin=min_radius, vmax=max_radius)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.1)
                cbar.set_label("Radius")

        # Add grid
        ax.grid(True)

        # Adjust tight layout
        plt.tight_layout()

        # Save figure if specified
        if viz_filename:
            plt.savefig(viz_filename, dpi=300, bbox_inches="tight")
            logging.info(f"Figure saved to {viz_filename}")

        # Show figure if specified
        if show:
            plt.show()

        return fig

    def _plot_sphere_packing_2d(
        self,
        centers: list[list[float]],
        radii: list[float],
        domain_size: tuple[float, float],
        porosity: float,
        show: bool,
        viz_filename: str,
        figsize=(10, 8),
    ) -> plt.Figure:
        """
        Plot 2D sphere packing visualization.

        Parameters:
        -----------
        centers: list
            List of all sphere centers

        radii: list
            List of all sphere radii

        domain_size: list
            List of all the dimensions

        Porosity: float
            Porosity of the structure. By default None

        show : bool
            Whether to show the plot. By default False

        save_path : str
            Path to save the figure (if None, figure is not saved). By default None

        figsize : tuple
            Figure size (width, height). By default (10,8)

        Returns:
        --------
        fig : matplotlib figure
            The figure object
        """

        # Initialize
        fig, ax = plt.subplots(figsize=figsize)

        # Define default values
        min_radius = 0
        max_radius = 1
        radius_range = 1
        cmap = mpl.colormaps["viridis"]

        # Define a colormap based on radii
        if len(radii) > 0:
            min_radius = np.min(radii)
            max_radius = np.max(radii)
            radius_range = max_radius - min_radius

            # Plot each circle
            for i, (center, radius) in enumerate(zip(centers, radii)):
                # Normalize radius for coloring
                if radius_range > 0:
                    color = cmap(Normalize(vmin=min_radius, vmax=max_radius)(radius))
                else:
                    color = cmap(0)

                circle = Circle((center[0], center[1]), radius, alpha=0.6, color=color)
                ax.add_patch(circle)

        # Set axis limits
        ax.set_xlim((0, domain_size[0]))
        ax.set_ylim((0, domain_size[1]))

        # Set aspect ratio
        ax.set_aspect("equal")

        # Add axis labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add title with statistics
        title = f"2D Circle Packing with {len(centers)} circles"
        if porosity is not None:
            title += f", Porosity of {100 * porosity:.4f}%"
        ax.set_title(title)

        # Add colorbar for radius
        if len(radii) > 0 and radius_range > 0:
            norm = Normalize(vmin=min_radius, vmax=max_radius)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20, pad=0.05)
            cbar.set_label("Radius")

        # Add grid
        ax.grid(True)

        # Adjust tight layout
        plt.tight_layout()

        # Save figure if specified
        if viz_filename:
            plt.savefig(viz_filename, dpi=300, bbox_inches="tight")
            logging.info(f"Sphere packing visualization saved to {viz_filename}")

        # Show figure if specified
        if show:
            plt.show()

        return fig

    def plot_size_distribution(
        self,
        radii: list[float],
        bins=20,
        figsize=(10, 6),
        show=False,
        directory=".",
        filename=None,
    ) -> plt.Figure:
        """
        Plot size distribution histogram.

        Parameters:
        -----------
        radii : numpy.ndarray
            Array of sphere radii

        bins : int
            Number of histogram bins

        figsize : tuple
            Figure size (width, height)

        show : bool
            Whether to show the plot

        directory: str
            Output directory

        filename : str
            Filename of the figure (if None, figure is not saved)

        Returns:
        --------
        fig : matplotlib figure
            The figure object
        """

        # Make filename path
        if filename is not None:
            viz_filename = os.path.join(directory, filename + "_size_distribution.png")
        else:
            viz_filename = None

        # Initialize figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histogram
        _, bins, _ = ax.hist(radii, bins=bins, density=True, alpha=0.7, color="skyblue")

        # Add statistics
        mean_radius = float(np.mean(radii))
        median_radius = float(np.median(radii))
        std_radius = float(np.std(radii))
        # Add vertical lines for mean radius and median radius
        ax.axvline(
            mean_radius,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {mean_radius:.4f}",
        )
        ax.axvline(
            median_radius,
            color="green",
            linestyle="dashed",
            linewidth=2,
            label=f"Median: {median_radius:.4f}",
        )

        # Add text box with statistics
        stats_text = f"Mean: {mean_radius:.4f}\nMedian: {median_radius:.4f}\nStd Dev: {std_radius:.4f}"
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        # Set axis labels
        ax.set_xlabel("Radius")
        ax.set_ylabel("Probability Density")

        # Add title
        ax.set_title(f"Porosities Size Distribution Histogram (n={len(radii)})")

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, alpha=0.3)

        # Adjust tight layout
        plt.tight_layout()

        # Save figure if specified
        if filename is not None:
            plt.savefig(viz_filename, dpi=300, bbox_inches="tight")
            logging.info(f"Size distribution visualization saved to {viz_filename}")

        # Show figure if specified
        if show:
            plt.show()

        return fig

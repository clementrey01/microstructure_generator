"""
========================================================================================================================
INPUT/OUTPUT UTILITY FUNCTIONS
========================================================================================================================
Description:
------------
This module contains the functions to input and output files
"""

import os
import json
import numpy as np
import logging
from configuration.config import StructureParams


def create_unique_output_path(
    directory_path: str,
    filename: str,
) -> str:
    """
    Create a unique output path if it doesn't already exist.

    Parameters:
    -----------
    directory_path : str
        Absolute path of the directory

    filename : str
        Name of the filename

    Returns:
    --------
    output_path : str
        Unique output path
    """
    # Expand the user's home directory in the directory_path
    expanded_directory_path = os.path.expanduser(directory_path)

    # Create the base output path
    output_path = os.path.join(expanded_directory_path, filename)

    # Check if the output path exists
    if not os.path.exists(output_path):
        # Create the directory if it does not exist
        os.makedirs(output_path)
        logging.info(f"Created output directory: {output_path}")
        return output_path
    else:
        # If the directory exists, find a unique name
        i = 1
        while True:
            unique_output_path = f"{output_path}_{i}"
            if not os.path.exists(unique_output_path):
                os.makedirs(unique_output_path)
                logging.info(f"Created output directory: {unique_output_path}")
                return unique_output_path
            i += 1


def save_metadata(
    metadata: dict,
    filename: str,
    directory: str,
) -> None:
    """
    Save metadata dictionary to JSON file.

    Parameters:
    -----------
    metadata: dict
        Metadata dictionary

    filename: str
        Output filename

    directory: str
        Output directory

    Returns:
    --------
    None
    """

    metadata_filename = os.path.join(directory, filename + "_metadata.json")

    # Convert numpy arrays to lists for JSON serialization
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            metadata[key] = value.tolist()

    # Save metadata to file
    with open(metadata_filename, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info(f"Saved metadata to {metadata_filename}")


def save_data(
    data: dict,
    filename: str,
    directory: str,
) -> None:
    """
    Save metadata dictionary to JSON file.

    Parameters:
    -----------
    data: dict
        Data dictionary

    filename: str
        Output filename

    directory: str
        Output directory

    Returns:
    --------
    None
    """

    data_filename = os.path.join(directory, filename + "_data.json")

    # Convert numpy arrays to lists for JSON serialization
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()

    # Save metadata to file
    with open(data_filename, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"Saved data to {data_filename}")


def load_metadata(filename: str) -> dict:
    """
    Load metadata dictionary from JSON file.

    Parameters:
    -----------
    filename : str
        Input filename

    directory_path : str
        Directory where the json is loaded. By default, working directory

    Returns:
    --------
    metadata : dict
        Metadata dictionary
    """
    with open(filename, "r") as f:
        metadata = json.load(f)
        if type(metadata) is not dict:
            raise TypeError("Metadata must be a dict.")

    logging.info(f"Loaded metadata from {filename}")
    return metadata


def load_params_from_file(
    filename: str,
    directory_path=".",
):
    """
    Load parameters from JSON file.

    Parameters:
    -----------
    filename : str
        Input filename

    Returns:
    --------
    params : StructureParams
        Parameters object
    """

    full_path = os.path.join(directory_path, filename)

    params = StructureParams.load(full_path)

    # Validate loaded parameters
    valid, message = params.validate()  # type: ignore
    if not valid:
        logging.error(f"Parameter validation failed: {message}")
        raise ValueError(f"Invalid parameters in {full_path}: {message}")

    return params

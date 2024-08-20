# ===========================================================================
#                            File Operation Helpers
# ===========================================================================

import json
import csv
import os


def find_csv_files(root_folder):
    """Find all CSV files in a folder and its subfolders."""
    csv_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            csv_files.append(os.path.join(dirpath, filename))

    return csv_files


def read_csv_as_dict(filename):
    """Read a CSV file and return the data as a list of dictionaries."""

    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    return data


def read_json_file(file_path: str) -> dict:
    """
    Read and parse a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file specified by `file_path` is not found.
        json.JSONDecodeError: If there is an error decoding the JSON data.
        Exception: If any other error occurs while reading the JSON file.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(
            f"Error: Failed to decode JSON in file '{file_path}'."
        )
    except Exception as e:
        raise Exception(
            f"Error: An error occurred while reading the JSON file: {str(e)}"
        )

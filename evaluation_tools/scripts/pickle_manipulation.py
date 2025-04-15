"""
This script defines the functions for manipulation with the pickle files.
"""

import pickle


def load_pickle_file(filename: str):
    """
    Loads pickle file.

    :param filename: Name of the pickle file.
    :return: Returns content of the pickle file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def store_pickle_file(filename: str, data_to_store):
    """
    Stored data to pickle file.

    :param filename: Filename.
    :param data_to_store: Data to be saved.
    """
    with open(filename, "wb") as f:
        pickle.dump(data_to_store, f)
    print(f"Data saved to {filename}")

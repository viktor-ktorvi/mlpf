import glob
import os
import pickle

from tqdm import tqdm
from typing import Any, Callable, List


def load_pickle_sample(filepath: str) -> Any:
    """
    The default way of loading a data sample(usually a PPC). Assumes that the file is a pickled object.
    :param filepath: Path to a file containing the pickled data sample.
    :return: Data sample.
    """
    with open(filepath, 'rb') as f:
        data_sample = pickle.load(f)

    return data_sample


def load_data(path: str, extension: str = ".p", load_sample_function: Callable = load_pickle_sample) -> List[Any]:
    """
    Load all the data files with the given extension in the directory described by path. By default, assumes that the files are pickled.
    :param path: Path to the data directory.
    :param extension: Pile extension.
    :param load_sample_function: Function to load the data samples. By default, is a function that loads a pickled object. The user can pass
    in their own function which takes a single string filepath argument.
    :return: List of the loaded data samples.
    """
    data_list = []
    for filepath in tqdm(glob.glob(os.path.join(path, "*" + extension)), ascii=True, desc="Loading files"):
        data_list.append(load_sample_function(filepath))

    return data_list

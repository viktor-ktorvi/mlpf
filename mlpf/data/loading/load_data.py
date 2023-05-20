import collections
import glob
import os
import pickle

from pandapower import from_json
from tqdm import tqdm
from typing import Any, Callable, Dict, List

from mlpf.data.utils.conversion import pandapower2ppc_list


def load_pickle_ppc(filepath: str) -> Dict:
    """
    The default way of loading a data sample(usually a PPC). Assumes that the file is a pickled object.

    :param filepath: Path to a file containing the pickled data sample.
    :return: Data sample.
    """
    with open(filepath, 'rb') as f:
        data_sample = pickle.load(f)

    assert type(data_sample) == dict, "Data sample isn't a dictionary which is the expected type for PyPower case files"

    return data_sample


def load_data(path: str, extension: str = ".p", load_sample_function: Callable = load_pickle_ppc) -> List[Any]:
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


def autodetect_load_ppc(path: str) -> List[Dict]:
    """
    Automatically detect the extension that appears in the given directory the most and load the files with that extension if supported.
    For more control over loading data use _load_data_.

    :param path: Data directory path.
    :return: List of PPCs
    """
    extension_counter = collections.Counter()
    for filename in glob.glob(os.path.join(path, "*")):
        name, ext = os.path.splitext(filename)
        extension_counter[ext] += 1

    most_common_extension = max(extension_counter, key=extension_counter.get)

    if most_common_extension == ".p":
        return load_data(path, extension=".p", load_sample_function=load_pickle_ppc)

    # TODO this could be decompositioned into it's own function.
    if most_common_extension == ".json":
        pandapower_networks = load_data(path, extension=".json", load_sample_function=from_json)

        return pandapower2ppc_list(pandapower_networks)

    raise ValueError(f"The most common extension found '{most_common_extension}' is not supported.")

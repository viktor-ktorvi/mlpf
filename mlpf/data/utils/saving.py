import glob
import os
import pickle

from pathlib import Path
from tqdm import tqdm
from typing import Any, List


def pickle_all(data_list: List[Any], save_path: str, extension: str = ".p", delete_all_from_save_path: bool = False):
    """
    Save all the objects in the data list as pickle files.

    :param data_list: List of picklable objects.
    :param save_path: Where to save the files.
    :param extension: Under what extension to save. Default value ".p"
    :param delete_all_from_save_path: If True everything from the save_path(if it exists) will be removed prior to saving. Default value False.
    :return:
    """
    Path(save_path).mkdir(exist_ok=True)

    if delete_all_from_save_path:
        # delete everything in the folder
        files = glob.glob(os.path.join(save_path, "*"))
        for f in tqdm(files, ascii=True, desc=f"Deleting all files from {save_path}"):
            os.remove(f)

    how_many = len(data_list)
    for i in tqdm(range(how_many), ascii=True, desc=f"Saving files to {save_path}"):
        with open(os.path.join(save_path, str(i).zfill(len(str(how_many))) + extension), 'wb') as f:
            pickle.dump(data_list[i], f)

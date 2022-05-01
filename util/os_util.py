import os
from os import walk


def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def files_in_dir(path):
    return next(walk(path), (None, None, []))[2]

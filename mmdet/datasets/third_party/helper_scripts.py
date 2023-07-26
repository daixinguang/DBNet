"""
Random scripts that just don't fit anywhere
"""

import os

from mmdet.datasets.third_party import dataset_constants


def get_files_from_folder(directory, extension=None):
    label_files = []
    for root, _, files in os.walk(directory):
        for some_file in files:
            label_files.append(os.path.abspath(os.path.join(root, some_file)))
    if extension is not None:
        label_files = list(filter(lambda x: x.endswith(extension), label_files))
    return label_files


def get_label_base(label_path):
    return '/'.join(label_path.split('/')[-2:])


def get_labels(split='test'):
    label_paths = get_files_from_folder(
        os.path.join(dataset_constants.LABELS, split), '.json')
    return label_paths


def ir(some_value):
    return int(round(some_value))

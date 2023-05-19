import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm

from utils import brute_force_matching, jpg_filename


FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features", "sift")


def get_most_similar_ids(im_array: ndarray, mapping_filename_to_id: DataFrame) -> List[int]:

    similarities_dict = get_similarities(im_array)
    best_matches = sorted(similarities_dict, key=lambda x:similarities_dict[x], reverse=True)[:3]

    ids = []
    for filename in best_matches:
        corresponding_id = mapping_filename_to_id.loc[mapping_filename_to_id.filename == filename, "id"].values[0]
        ids += [int(corresponding_id)]
    return ids


def get_similarities(im_array: ndarray, features_dir: Path = FEATURES_DIR) -> Dict[str, float]:

    # get microsoft beit features
    image_features = get_features(im_array)
    
    # get list of descriptors filenames
    filenames = os.listdir(features_dir)

    similarity = {}
    for filename in tqdm(filenames):

        # get path
        sample_features_path = os.path.join(features_dir, filename)

        # load descriptors
        sample_features = np.load(sample_features_path)

        # get number of matches
        similarity[jpg_filename(filename)] = brute_force_matching(image_features, sample_features)
    
    return similarity


def get_features(im_array: ndarray) -> ndarray:
    sift = cv2.SIFT_create()
    _, features = sift.detectAndCompute(im_array, None)
    return features

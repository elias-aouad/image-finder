import os
from typing import Dict, List
from pathlib import Path

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from PIL.Image import Image
from pandas import DataFrame
from transformers import BeitImageProcessor, BeitForImageClassification

from utils import brute_force_matching, jpg_filename


FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features", "microsoft_beit")

processor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224')
model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')


def get_most_similar_ids(image: Image, mapping_filename_to_id: DataFrame) -> List[int]:

    similarities_dict = get_similarities(image)
    best_matches = sorted(similarities_dict, key=lambda x:similarities_dict[x], reverse=True)[:3]

    ids = []
    for filename in best_matches:
        corresponding_id = mapping_filename_to_id.loc[mapping_filename_to_id.filename == filename, "id"].values[0]
        ids += [int(corresponding_id)]
    return ids


def get_similarities(image: Image, features_dir: Path = FEATURES_DIR) -> Dict[str, float]:

    # get microsoft beit features
    image_features = get_features(image)
    
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


def get_features(image: Image) -> ndarray:
    pixel_values = processor([image], return_tensors="pt").pixel_values
    output = model(pixel_values, output_hidden_states=True)
    pooled_output = output.hidden_states[-1][0, :].data.numpy()
    return pooled_output

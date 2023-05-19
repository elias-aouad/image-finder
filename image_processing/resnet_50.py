import os
from typing import Dict, List
from pathlib import Path

import numpy as np
from numpy import ndarray
from tqdm import tqdm
from PIL.Image import Image
from pandas import DataFrame

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from utils import cosine_similarity, jpg_filename


FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "features", "resnet_50")

# define model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# define image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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

        # get similarity
        similarity[jpg_filename(filename)] = cosine_similarity(image_features, sample_features)
    
    return similarity


def get_features(image: Image) -> ndarray:
    pixel_values = transform(image).unsqueeze(0)
    features = model(pixel_values)
    return features.flatten().data.numpy()

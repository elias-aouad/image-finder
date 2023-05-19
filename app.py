import os

import cv2
import numpy as np
from pandas import read_csv
from PIL import Image
from flask import Flask, request, jsonify

from image_processing import microsoft_beit, resnet_50, sift


CSV_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
FILENAME_TO_ID = read_csv(CSV_PATH)
FILENAME_TO_ID["filename"] = FILENAME_TO_ID["url"].apply(lambda url: url.split("/")[-1])

app = Flask(__name__)


@app.route('/resnet50-similar-images', methods=['POST'])
def resnet50_find_similar_images():
    # Access the uploaded image from the request
    file = request.files['image']
    im_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(im_array)

    # get similar images
    similar_image_ids = resnet_50.get_most_similar_ids(image, FILENAME_TO_ID)
    return jsonify(similar_image_ids)


@app.route('/beit-similar-images', methods=['POST'])
def beit_find_similar_images():
    # Access the uploaded image from the request
    file = request.files['image']
    im_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = Image.fromarray(im_array)

    # get similar images
    similar_image_ids = microsoft_beit.get_most_similar_ids(image, FILENAME_TO_ID)
    return jsonify(similar_image_ids)


@app.route('/sift-similar-images', methods=['POST'])
def sift_find_similar_images():
    # Access the uploaded image from the request
    file = request.files['image']
    im_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # get similar images
    similar_image_ids = sift.get_most_similar_ids(im_array, FILENAME_TO_ID)
    return jsonify(similar_image_ids)


if __name__ == '__main__':
    app.run()

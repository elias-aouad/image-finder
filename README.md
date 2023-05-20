# Image finder

Hi, this is a repository that implements an image matching API.


## How to setup

First, create a virtual environment and make sure you're using python 3.8

Second, follow these commands :

- Step 1 :  Install python requirements

```commandline
pip install -r dev.requirements.txt
```

- Step 2 : Download extracted features

```commandline
bash download_features.sh
```

- Step 3 : Run API

```commandline
python app.py
```


## Run a request

Once you've followed the setup instructions, you can make requests to the API.

To be noted : Three different methodologies are implemented for finding similarities

- SIFT

```commandline
curl -X POST -F "image=@/path/to/image.jpg" http://172.17.0.2:5000/sift-similar-images
```

- ResNet-50

```commandline
curl -X POST -F "image=@/path/to/image.jpg" http://172.17.0.2:5000/resnet50-similar-images
```

- Microsoft Beit model

```commandline
curl -X POST -F "image=@/path/to/image.jpg" http://172.17.0.2:5000/beit-similar-images
```

PS : You can find some samples in the folder `example_images`


## Docker

You can also run the API on a docker container for more robustness. Follow these steps :

- Build docker image :

```commandline
sudo docker build -t image-finder .
```

- Run it :

```commanline
sudo docker run -p 5000:5000 image-finder
```

## Methodology

- Step 1 : Feature extraction

  This project relies on matching images. 

  Naturally, we need to extract features from raw images, and I did it using different methods :

    - SIFT features

    - ResNet-50 features

    - Microsoft Beit features

  PS : The extraction was done via Colab notebooks. Find all necessay details <a href="https://drive.google.com/drive/folders/1ZWhsn1-76QYfPqoPLZn7Ms_g_1hrq0b_?usp=sharing" target="_blank">here</a>.

  The features were then saved in `.npy` format for each image in the dataset.


- Step 2 : Matching metrics

  Different methodologies were used for matching.

  For ResNet-50, simple cosine similarity does the trick.

  However, for the SIFT and microsoft-Beit features, since they extract local features, we cannot rely on cosine similarity.

  Hence, global matching between features of image A and image B were used. Here is how it works :

      - Say image A has `n` local features and image B has `m` local features, and a latent space of dimension `h`

      - The dot product between features matrix A and B provides a matrix of size `n` x `m`

      - Each elements (i, j) of this matrix provides information of the matching similarity between `i`-th feature of 
      matrix A and `j`-th feature of matrix B.

      - We count all matching features (i, j) as all the pairs of features whose most similar with each other, which means

          - Feature `i` is the most similar to feature `j` AND

          - Feature `j` is the most similar to feature `i`

  This is referred to brute force matching in the code (can be found in `utils.py`)

- Step 3 :

  Once a similarity metric is defined for each pair of images, and since we already extracted features from the images provided in the dataset,
  when a new image is provided, here are the steps that goes through the API :

  1. Extract features from new images

  2. Compute similarity to all already extracted features

  3. Find the 3 highest similarity results

  4. Output the corresponding image ids (in order)


## Conclusion

This methodology provides differents ways of extracting/matching features from images.

The one that worked the most was ResNet-50, even if the result is not always perfect.

When using deep learning approaches, one should always pay attention to the data on which the model was pretrained.

This data may lead to interpretations of how the model extracts features from images.

If we had more data, few-shot learning would be a good training scheme for our usage.

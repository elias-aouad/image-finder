# Image finder

Hi, this is a repository that implements an image matching API.


## How to setup

First, ceate a virtual environment and make sure you're using python 3.8

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

To be noted : Three different methods are used for finding similarities : 

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

You can also run API on a docker container for more robustness. Follow these steps :

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

      - We count all matching features (i, j) as all the pair of features whose most similar with each other, which means

          - Feature `i` is the most similar to feature `j` AND

          - Feature `j` is the most similar to feature `i`
          
  This is referred to brute force matchng in the code (you can find it in `utils.py`)




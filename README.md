# Image finder

Hi, this is a repository that implements an image matching API.


## How to setup

First, make sure you're using python 3.8

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




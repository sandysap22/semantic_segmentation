# Semantic Segmentation
## Author: Sandeep Patil

[vgg]: ./readme_images/vgg.png "vgg"
[demo]: ./readme_images/demo.gif "demo"

### Overview
Objective of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN). We have extended VGG16 network to do segmentation task.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![demo][demo]  

### The Model
The VGG model can be downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![vgg][vgg]  

This is customized VGG in which last fully connected layer is repalced with 1x1x4096 convolution layer.
We upsample this model to produce output of 160x576x2 to get final logits which represents masking for segmentation. 

The sample output can be found [here](./result_images) 

### Setup
##### GPU
For GPU you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


##### Run
Run the following command to run the project:
```
python main.py
```

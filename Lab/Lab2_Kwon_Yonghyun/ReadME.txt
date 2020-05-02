Version used:
Python 3.7.4

The name and version of additional packages:
Numpy:  1.16.5
TensorFlow: 2.0.0
Keras: 2.2.4-tf
pandas: 0.25.1

How to run my code in Windows command line:
python main.py

How to change the parameters (listed in task (a) in the project description) in your code, including files and lines.
To perform fully-connected feed-forward neural network, in feed_forward.py, change input parameters and run feed_forward.py
3rd line:
# Input parameters
loss = "mse"
activation = "relu"
scale1, nHiddenlayers1, nHiddenunits1, lr1, momentum1, batch_size1 = [1, 2, 1000, 0.1, 0.0, 8]

To perform convolutional neural network, in convolution.py, change input parameters and run convolution.py
3rd line:
# Input parameters
filtersize5, nconv5, height5, pool_size5 = [20, 2, 2, 1]
batch_size5, lr5, momentum5 = [32, 0.1, 0.0]
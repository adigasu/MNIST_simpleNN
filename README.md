# MNIST_simpleNN
MNIST digit classification using simple neural network (NN) using python (without using any deep learning library)

Simple 3 layer fully connected NN - [256, 64, 10] hidden units

Loss function - Categorical cross entropy loss

Optimizer - Stochastic gradient descent (SGD) with intial learning rate of 0.1 and exponential decay factor 1e-8

Accuracy - Average accuracy (5 trails): 97.62 %, Highest accuracy: 97.69 %

Time - It takes around 33 minutes for 30 epochs, however it convergence between 10 to 15 epochs

### Dependencies
This code depends on the following libraries:

- numpy
- matplotlib
- tqdm

Code should be compatible with any version of python. (tested in python2.7)

### Training

The network can be trained to reproduced with command:  
```
python train.py
```
The MNIST data is placed in following path: "./data/"

### TODO
- Generalize the code to take inputs from command line 
- Add nesterov momentum for SGD
- Check the performance with Adam optimizer
- Implementation of Convolution layer

### Reference
- [The Softmax function and its derivative](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/#)

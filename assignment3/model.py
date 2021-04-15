import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers

        self.conv1 = ConvolutionalLayer(\
                                        in_channels = input_shape[-1],\
                                        out_channels = conv1_channels,\
                                        filter_size=3,\
                                        padding=1
                                       )
        self.relu1 = ReLULayer()
        self.max_pool_1 = MaxPoolingLayer(\
                                          pool_size = 4,
                                          stride = 4
                                         )
        self.conv2 = ConvolutionalLayer(\
                                        in_channels = conv1_channels,\
                                        out_channels = conv2_channels,\
                                        filter_size=3,\
                                        padding=1)
        self.relu2 = ReLULayer()
        self.max_pool_2 = MaxPoolingLayer(\
                                          pool_size = 4,
                                          stride = 4
                                         )
        self.flatten = Flattener()
        self.fc = FullyConnectedLayer(n_input=8, n_output=n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        
        (self.conv1.params()['W']).grad = np.zeros_like((self.conv1.params()['W']).grad)
        (self.conv1.params()['B']).grad = np.zeros_like((self.conv1.params()['B']).grad) 
        (self.conv2.params()['W']).grad = np.zeros_like((self.conv2.params()['W']).grad)
        (self.conv2.params()['B']).grad = np.zeros_like((self.conv2.params()['B']).grad)
        (self.fc.params()['W']).grad = None
        (self.fc.params()['B']).grad = None
        
        conv1 = self.conv1.forward(X)
        relu1 = self.relu1.forward(conv1)
        pool_1 = self.max_pool_1.forward(relu1)
        conv2 = self.conv2.forward(pool_1)
        relu2 = self.relu2.forward(conv2)
        pool_2 = self.max_pool_2.forward(relu2)
        flat = self.flatten.forward(pool_2)
        fc = self.fc.forward(flat)
        fc -= np.max(fc, axis=1)[:, None]
        
        loss, grad = softmax_with_cross_entropy(fc, y)
        
        d_fc = self.fc.backward(grad)
        d_flat = self.flatten.backward(d_fc)
        d_pool_2 = self.max_pool_2.backward(d_flat)
        d_relu2 = self.relu2.backward(d_pool_2)
        d_conv2 = self.conv2.backward(d_relu2)
        d_pool_1 = self.max_pool_1.backward(d_conv2)
        d_relu1 = self.relu1.backward(d_pool_1)
        d_conv1 = self.conv1.backward(d_relu1)
        
        return loss
        

    def predict(self, X):
        # You can probably copy the code from previous assignment
        
        pred = np.zeros(X.shape[0], np.int)
        
        conv1 = self.conv1.forward(X)
        relu1 = self.relu1.forward(conv1)
        pool_1 = self.max_pool_1.forward(relu1)
        conv2 = self.conv2.forward(pool_1)
        relu2 = self.relu2.forward(conv2)
        pool_2 = self.max_pool_2.forward(relu2)
        flat = self.flatten.forward(pool_2)
        fc = self.fc.forward(flat)

        fc -= np.max(fc, axis=1)[:, None]
        
        e_probs = np.exp(fc)
        sum_probs = np.sum(e_probs, axis=1)[:, None]
        probs = e_probs/sum_probs
        
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        
        result = {'W1': self.conv1.params()['W'], 'B1':self.conv1.params()['B'],
                 'W2': self.conv2.params()['W'], 'B2':self.conv2.params()['B'],
                 'W3': self.fc.params()['W'], 'B3':self.fc.params()['B']}

        return result

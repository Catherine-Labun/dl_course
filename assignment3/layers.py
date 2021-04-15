import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    
    return loss, grad

def softmax(predictions):
    
    pred = predictions.copy()
    pred -= np.max(pred, axis=1)[:, None]
    e_pred = np.exp(pred)
    sum_pred = np.sum(e_pred, axis=1)[:, None]
    probs = e_pred/sum_pred
    
    return probs


def cross_entropy_loss(probs, target_index):
    
    batch_size = len(target_index)
    prob = np.zeros(batch_size)
    prob = probs[range(batch_size),target_index]
    loss = np.sum(-np.log(prob))
    
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    
    S = softmax(predictions)
    loss = cross_entropy_loss(S, target_index)
    d_preds = S
    d_preds[range(d_preds.shape[0]), target_index] -= 1

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X.copy()
        relu = np.maximum(0, self.X)
        
        return relu

    def backward(self, d_out):
        
        self.X[self.X>0] = 1
        self.X[self.X<=0] = 0
        
        d_result = d_out*self.X
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        
        self.X = X.copy()
        
        forward = np.dot(self.X, self.W.value) + self.B.value
        
        return forward

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)    
        
        self.B.grad = np.sum(d_out, axis = 0)[None, :]
        
        d_input = np.dot(d_out, self.W.value.T)

        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.padding = padding


    def forward(self, X):
        
        self.X = X.copy()
        batch_size, height, width, channels = X.shape

        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        self.X = np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values = (0))
        
        out_height = height + 2*self.padding - self.filter_size + 1
        out_width = width + 2*self.padding - self.filter_size + 1       
        output = np.zeros((batch_size, out_height, out_width, self.out_channels))
 
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        
        for y in range(out_height):
            for x in range(out_width):
                output[:, y, x] = np.dot(self.X[:, y:y+self.filter_size, x:x+self.filter_size].reshape(batch_size, -1),\
                                   self.W.value.reshape(-1, self.out_channels)) + self.B.value
         
        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_input = np.zeros_like(self.X)
        # Try to avoid having any other loops here too
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                self.W.grad += (np.dot(self.X[:, y:y+self.filter_size, x:x+self.filter_size].reshape(batch_size, -1).T,\
                                       d_out[:, y, x])).reshape(self.W.grad.shape)
                self.B.grad += np.sum(d_out[:, y, x], axis = 0)
                
                d_input[:, y:y+self.filter_size, x:x+self.filter_size] += np.dot(d_out[:, y, x],\
                                                                                 self.W.value.reshape(-1, self.out_channels).T)\
                                                                          .reshape(batch_size, self.filter_size, self.filter_size, -1)
        
        d_input = d_input[:, self.padding:height-self.padding, self.padding:width-self.padding]
                
        return d_input
        

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        self.X = X.copy()
        batch_size, height, width, channels = self.X.shape
        
        out_height = (height-self.pool_size)//self.stride + 1
        out_width = (width-self.pool_size)//self.stride + 1
        
        d_out = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                d_out[:, y, x] = self.X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size]\
                                                         .reshape(batch_size, -1, channels).max(axis=1) 
                
        return d_out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                mask = self.X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size]\
                       == self.X[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size]\
                                                         .reshape(batch_size, -1, channels).max(axis=1)[:, None, None]
                
                d_input[:, y*self.stride:y*self.stride+self.pool_size, x*self.stride:x*self.stride+self.pool_size] = mask\
                *d_out[:, y, x][:, None, None]
        
        return d_input
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, _, _, _ = X.shape
        
        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        
        return X.reshape(batch_size, -1)

    def backward(self, d_out):
        # TODO: Implement backward pass
        
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

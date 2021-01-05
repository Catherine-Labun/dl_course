import numpy as np


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
    prob = probs[range(batch_size),target_index.ravel()]
    loss = np.mean(-np.log(prob))
    
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment

    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    
    S = softmax(preds)
    loss = cross_entropy_loss(S, target_index)
    p = np.zeros_like(S)
    batch_size = len(target_index)
    p[range(batch_size), target_index.ravel()] = 1
    d_preds = (S - p)/batch_size

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        self.X = X.copy()
        relu = np.maximum(0, self.X)
        
        return relu

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        
        self.X[self.X>0] = 1
        self.X[self.X<=0] = 0
        
        d_result = d_out*self.X
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        
        self.X = X.copy()
        
        forward = np.dot(self.X, self.W.value) + self.B.value
        
        return forward
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)    
        
        self.B.grad = np.sum(d_out, axis = 0)[None, :]
        
        d_result = np.dot(d_out, self.W.value.T)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}

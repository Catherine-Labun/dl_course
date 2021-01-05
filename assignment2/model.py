import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.w1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.w2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu = ReLULayer()
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        (self.w1.params()['W']).grad = None
        (self.w1.params()['B']).grad = None 
        (self.w2.params()['W']).grad = None
        (self.w2.params()['B']).grad = None
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        w1 = self.w1.forward(X)
        relu = self.relu.forward(w1)
        w2 = self.w2.forward(relu)
        loss, grad = softmax_with_cross_entropy(w2, y)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        loss_w1, grad_w1 = l2_regularization((self.w1.params()['W']).value, self.reg)
        loss_w2, grad_w2 = l2_regularization((self.w2.params()['W']).value, self.reg)
        
        loss_b1, grad_b1 = l2_regularization((self.w1.params()['B']).value, self.reg)
        loss_b2, grad_b2 = l2_regularization((self.w2.params()['B']).value, self.reg)
        
        loss += loss_w1 + loss_w2 + loss_b1 + loss_b2
        
        d_w2 = self.w2.backward(grad)
        d_relu = self.relu.backward(d_w2)
        d_w1 = self.w1.backward(d_relu)
        
        (self.w1.params()['W']).grad += grad_w1
        (self.w1.params()['B']).grad += grad_b1
        
        (self.w2.params()['W']).grad += grad_w2
        (self.w2.params()['B']).grad += grad_b2
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        w1 = self.w1.forward(X)
        relu = self.relu.forward(w1)
        w2 = self.w2.forward(relu)

        w2 -= np.max(w2, axis=1)[:, None]
        e_probs = np.exp(w2)
        sum_probs = np.sum(e_probs, axis=1)[:, None]
        probs = e_probs/sum_probs
        
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        result = {'W1': self.w1.params()['W'], 'B1':self.w1.params()['B'],
                 'W2': self.w2.params()['W'], 'B2':self.w2.params()['B']}

        return result

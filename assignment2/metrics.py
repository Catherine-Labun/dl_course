import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    
    accuracy = 0
    num_test = prediction.shape[0]
    tp = np.sum(prediction == ground_truth)
    accuracy = tp/num_test
    
    return accuracy

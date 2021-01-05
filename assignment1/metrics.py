import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    tp = list(ground_truth&prediction).count(1)
    fp = list((ground_truth ^ prediction)^ground_truth).count(1) - tp
    fn = list(ground_truth ^ prediction).count(1) - fp
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    accuracy = (list(prediction == ground_truth).count(1))/prediction.shape[0]
    
    f1 = 2 * precision * recall / (precision + recall) 
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = 0
    num_test = prediction.shape[0]
    tp = np.sum(prediction == ground_truth)
    accuracy = tp/num_test
    
    return accuracy

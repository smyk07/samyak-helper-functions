"""
A series of helper function I regularly use in my projects.
Many taken from `mrdbourke/pytorch-deep-learning`
"""

# Import all dependencies 
import torch 
from torch import nn 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# Define function to set all seeds 
def set_seeds(num: int): 
    """
    Sets `torch.manual_seed()` and `torch.cuda.manual_seed()` to the given `num: int` argument.
    """
    torch.manual_seed(num)
    torch.cuda.manual_seed(num) 

# Set device 
def set_device(variable = device):  
    """
    Functionizing simple device agnostic code by taking in the `device` variable as an argument.
    """
    variable = "cuda" if torch.cuda.is_available() else "cpu" 

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc 

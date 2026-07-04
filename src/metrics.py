"""
Unified MAPE calculation — tek kaynak, tüm modüller buradan import etmeli.
"""
import numpy as np


def calculate_mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.
    epsilon ile sıfır bölme korunur.
    """
    epsilon = 1e-10
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return mape

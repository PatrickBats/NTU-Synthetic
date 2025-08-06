import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()

def softmax_with_temperature(x, temperature=1.0):
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()

def scaled_softmax(x, scale=0.5):
    softmax_values = softmax(x)
    return scale * softmax_values + (1 - scale) * (x - x.min()) / (x.max() - x.min())

def sparsemax(x):
    sorted_x = np.sort(x)[::-1]
    cumsum = np.cumsum(sorted_x)
    k = np.arange(1, len(x) + 1)
    k_selected = k[cumsum - k * sorted_x > -1][-1]
    tau = (cumsum[k_selected - 1] - 1) / k_selected
    return np.maximum(x - tau, 0)
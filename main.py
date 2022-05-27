import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


def dif_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))



def loss(input, output):

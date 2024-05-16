#coding:utf-8
import numpy as np


class ActivateFunction:
    def __call__(self, x):
        raise NotImplementedError
    
    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(ActivateFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)
    
class ReLU(ActivateFunction):
    def __call__(self, x):
        x[x < 0] = 0
    
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
def softmax(x:np.ndarray):
    exp_x:np.ndarray = np.exp(x - np.max(x, axis=0))
    rate_x = exp_x / np.sum(exp_x, axis=0)
    return rate_x
    

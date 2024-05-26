#coding:utf-8
import numpy as np

def debugnpload():
    weight_matrix = []
    with np.load("hxnumocr\kd_config\weights.npz") as weight_load:            
        for i in range(len(weight_load)):
            windex = "weight_" + str(i + 1)
            weight_matrix.append(weight_load[windex].copy())
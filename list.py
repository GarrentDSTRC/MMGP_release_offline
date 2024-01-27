import math
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import gpytorch
from matplotlib import pyplot as plt
from pyKriging.samplingplan import samplingplan
import pandas as pd
from time import time
from scipy.interpolate import griddata
import os
import numpy as np
import random
from GPy import *
testmode="experiment"#DTLZ#WFG
path1 = r".\Database\train_x_H.csv"
path2 = r".\Database\y.csv"
X = np.loadtxt(path1, delimiter=',')


UPBound = np.array(UPB).T
LOWBound = np.array(LOWB).T
normalizer = Normalizer(LOWBound, UPBound)
X=normalizer.denormalize(X)
for i in range(0, len(X), 8):
    batch_X = X[i:i+8]
    initialDataX, initialDataY = findpointOL(batch_X, num_task=2, mode=testmode)
    if i<1:
       ALLY=initialDataY 
    else:
       ALLY=np.concatenate((ALLY,initialDataY),axis=0)

    # Save all y values to a CSV file
    np.savetxt(path2, ALLY, delimiter=',')









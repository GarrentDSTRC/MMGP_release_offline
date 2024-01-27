import numpy as np
import os
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt


UPB=[1.0, 0.6, 40, 180, 1000]
LOWB=[0.6, 0.1, 5, -180, 200]
import torch
class Normalizer:
    def __init__(self, low_bound=LOWB, up_bound=UPB):
        self.low_bound = torch.tensor(low_bound, dtype=torch.float32)
        self.up_bound = torch.tensor(up_bound, dtype=torch.float32)

    def normalize(self, x):
        x=torch.as_tensor(x)
        return (x - self.low_bound) / (self.up_bound - self.low_bound)

    def denormalize(self, norm_x):
        norm_x = torch.as_tensor(norm_x)
        return norm_x * (self.up_bound - self.low_bound) + self.low_bound
   #[0.972000002861023, 0.5400000214576721, 26.700000762939453, 32.399993896484375, 808.0]
norm=Normalizer()
#0.93,	0.88,	0.62	,0.59,	0.76
x=[0.84,	0.77,	0.74,	0.54,	0.87]
X=norm.denormalize(x).tolist()
print(X)

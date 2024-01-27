import pandas as pd
import numpy as np
from scipy.stats import norm
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
UPB=[1.0,0.6,40,180,9,9,35]
LOWB=[0.6,0.1,5,0,0,0,10]
# 从Excel中读取数据
data = pd.read_csv('train_x.csv')
data2 = pd.read_csv('train_y.csv')
# 获取输入和输出数据
inputs = data.iloc[:, 0:7].values
output1 = data2.iloc[:, 0].values
output2 = data2.iloc[:, 1].values
# 定义模型函数
def model(inputs):
    x1, x2, x3, x4, x5, x6, x7 = inputs
    y1 = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 6*x6 + 7*x7
    y2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2 + x7**2
    return y1, y2
# 定义输入参数和输出参数
problem = {
    'num_vars': 7,
    'names': ['input1', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7'],
    'bounds': list(zip(LOWB,UPB))
}
param_values = saltelli.sample(problem,1024)

Y = output1
Z = output2
# 执行敏感性分析
Si_Y = sobol.analyze(problem, Y)
Si_Z = sobol.analyze(problem, Z)
# 输出结果
print('Sensitivity analysis for output1:')
print(Si_Y)
print('Sensitivity analysis for output2:')
print(Si_Z)
# 可视化结果
fig, ax = plt.subplots(2, figsize=[6, 8])
ax[0].bar(problem['names'], Si['ST'])
ax[0].set_title('Total effect indices')
ax[1].bar(problem['names'], Si['S1'])
ax[1].set_title('First order indices')
plt.tight_layout()
plt.show()
import csv
# 读取csv文件中的数据
with open('data.csv', 'r') as file:
    data=[]
    reader = csv.reader(file)
    for row in reader:
        row = [float(i) for i in row]
        data.append(row)
# 将前7维数据作为输入，后2维数据作为输出
inputs = [row[:7] for row in data]
outputs = [row[7:] for row in data]
# 找出重复的输入数据
unique_inputs = []
unique_outputs = []
for i in range(len(inputs)):
    if inputs[i] not in unique_inputs:
        # 如果输入数据不重复，则将其添加到去重后的列表中
        unique_inputs.append(inputs[i])
        unique_outputs.append(outputs[i])
    else:
        # 如果输入数据重复，则将其对应的输出数据进行平均值计算，并更新去重后的列表中的数据
        index = unique_inputs.index(inputs[i])
        unique_outputs[index] = [(unique_outputs[index][j] + outputs[i][j]) / 2 for j in range(2)]
# 将去重后的数据输出到csv文件中
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(unique_inputs)):
        row = unique_inputs[i] + unique_outputs[i]
        writer.writerow(row)

import numpy as np
import csv
# 读取csv文件中的数据
with open('output.csv', 'r') as file:
    reader = csv.reader(file)
    data = []
    reader = csv.reader(file)
    for row in reader:
        row = [float(i) for i in row]
        data.append(row)

    inputs = [row[:7] for row in data]
    outputs = [row[7:] for row in data]
    # 将数据转换为numpy数组
    data = np.array(inputs, dtype=np.float32)
    # 将numpy数组保存为npy文件
    np.save('saveX_gpytorch_multi_EI_MS.npy', data)
    data = np.array(outputs, dtype=np.float32)
    # 将numpy数组保存为npy文件
    np.save('savey_gpytorch_multi_EI_MS.npy', data)
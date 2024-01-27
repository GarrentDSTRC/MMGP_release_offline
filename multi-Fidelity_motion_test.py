import math
import numpy as np
import torch
import gpytorch
import time
import os
from GPy import *
# path1=r".\RAM\saveX_gpytorch_multifidelity_multitask.npy"
# path2=r".\RAM\saveI_gpytorch_multifidelity_multitask.npy"
# path3=r".\RAM\saveY_gpytorch_multifidelity_multitask.npy"
# path4=r".\RAM\saveTestXdict_gpytorch_multifidelity_multitask.npy"
d=650
path1=r".\ROM\E2\saveX_gpytorch_multifidelity_multitask %d.npy"% (d)
path2=r".\ROM\E2\saveI_gpytorch_multifidelity_multitask %d.npy"% (d)
path3=r".\ROM\E2\saveY_gpytorch_multifidelity_multitask %d.npy"% (d)
path4=r".\ROM\E2\TestXdict_gpytorch_multifidelity_multitask %d.npy"% (d)
path5=r".\Database\saveMAE_MM.csv"

pathM='.\Database\model_state_MFM%d.pth'% (d)
# We make an nxn grid of training points spaced every 1/(n-1) on [0,1]x[0,1]
# n = 250
n = 30
n1=50
UpB=2200
LowB=0
testsample=140
training_iterations =70#68#35#68#48 #100
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device( "cpu")
num_tasks=-2
SCALE=0.1

# Construct data
if os.path.exists(path1):
    full_train_x = torch.FloatTensor(np.load(path1,allow_pickle=True))
    full_train_i = torch.FloatTensor(np.load(path2,allow_pickle=True))
    full_train_y = torch.FloatTensor(np.load(path3,allow_pickle=True))
    dict = np.load(path4, allow_pickle=True).astype(int).tolist()
    #np.savetxt("SAVEXMM.csv", full_train_x, delimiter=',')



likelihood1 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model1 = MultiFidelityGPModel((full_train_x, full_train_i), full_train_y[:,0], likelihood1).to(device)
likelihood2 = gpytorch.likelihoods.GaussianLikelihood().to(device)
model2 = MultiFidelityGPModel((full_train_x, full_train_i), full_train_y[:,1], likelihood2).to(device)
model = gpytorch.models.IndependentModelList(model1, model2).to(device)
likelihood = gpytorch.likelihoods.LikelihoodList(model1.likelihood, model2.likelihood)

print(model)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
from gpytorch.mlls import SumMarginalLogLikelihood
mll = SumMarginalLogLikelihood(likelihood, model)

if os.path.exists(pathM):
    state_dict = torch.load(pathM)
    model.load_state_dict(state_dict)
else:
    for j in range(training_iterations):
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward(retain_graph=True)
        print('Iter %d/%d - Loss: %.3f' % (j + 1, training_iterations, loss.item()))
        optimizer.step()
    torch.save(model.state_dict(), pathM)

#print("rho",model.task_covar_module1.rho)
#plot3D(model, likelihood , num_task=num_tasks,scale= [1,0.1]*10)

# Set into eval mode
model.eval()
likelihood.eval()

select=dict

test_x = torch.tensor(Frame.iloc[select, 0:4].to_numpy()).to(torch.float32)
test_xo=torch.tensor(Frame.iloc[:, 0:4].to_numpy()).to(torch.float32)

test_y_actualLC =torch.tensor(Frame.iloc[select, 6].to_numpy()).to(torch.float32)
test_y_actualLE = torch.tensor(Frame.iloc[select, 7].to_numpy()).to(torch.float32)
test_y_actualHC =torch.tensor(Frame.iloc[select, 4].to_numpy()).to(torch.float32)
test_y_actualHE = torch.tensor(Frame.iloc[select, 5].to_numpy()).to(torch.float32)

test_y_actualoHC =torch.tensor(Frame.iloc[:, 4].to_numpy()).to(torch.float32)
test_y_actualoHE = torch.tensor(Frame.iloc[:, 5].to_numpy()).to(torch.float32)

test_i_task1 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=0)
test_i_task2 = torch.full((test_x.shape[0], 1), dtype=torch.long, fill_value=1)

test_i_tasko = torch.full((test_xo.shape[0], 1), dtype=torch.long, fill_value=1)

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_yL = likelihood(*model((test_x, test_i_task1), (test_x, test_i_task1)))
    observed_pred_yH = likelihood(*model((test_x, test_i_task2), (test_x, test_i_task2)))
    observed_pred_yLC = observed_pred_yL[0].mean #ct low
    observed_pred_yLE = observed_pred_yL[1].mean #eta low
    observed_pred_yHC = observed_pred_yH[0].mean #ct high
    observed_pred_yHE = observed_pred_yH[1].mean #eta high

    observed_pred_yo = likelihood(*model((test_xo, test_i_tasko), (test_xo, test_i_tasko)))
    observed_pred_yoC =torch.max(observed_pred_yo[0].mean)#ct high
    observed_pred_yoE =torch.max(observed_pred_yo[1].mean) #eta high
    observed_pred_yoM = torch.max(0.5 * observed_pred_yo[0].mean + 0.5 * observed_pred_yo[1].mean*4)
    observed_pred_yoMindex=torch.argmax(0.5 * observed_pred_yo[0].mean + 0.5 * observed_pred_yo[1].mean*4)
    M1Raw=torch.tensor(Frame.iloc[observed_pred_yoMindex.item(), 0:6].to_numpy().astype(float)).to(torch.float32)

    observed_pred_yoM2 = torch.max(0.2 * observed_pred_yo[0].mean + 0.8 * observed_pred_yo[1].mean*4)
    observed_pred_yoM3 = torch.max(0.8 * observed_pred_yo[0].mean + 0.2 * observed_pred_yo[1].mean*4)
    observed_pred_yoM3index = torch.argmax(0.8 * observed_pred_yo[0].mean + 0.2 * observed_pred_yo[1].mean * 4)
    M3Raw = torch.tensor(Frame.iloc[observed_pred_yoM3index.item(), 0:6].to_numpy().astype(float)).to(torch.float32)

    observed_pred_yoM4=torch.max(0.9*observed_pred_yo[0].mean+0.1*observed_pred_yo[1].mean*4)
    observed_pred_yoM4index = torch.argmax(0.9*observed_pred_yo[0].mean+0.1*observed_pred_yo[1].mean*4)
    M4Raw = torch.tensor(Frame.iloc[observed_pred_yoM4index.item(), 0:6].to_numpy().astype(float)).to(torch.float32)

    observed_pred_yoM5 = torch.max(0.1 * observed_pred_yo[0].mean + 0.9 * observed_pred_yo[1].mean * 4)
    observed_pred_yoM6 = torch.max(-torch.abs( observed_pred_yo[0].mean-15)+  observed_pred_yo[1].mean * 4)
    observed_pred_yoM6index = torch.argmax(-torch.abs(observed_pred_yo[0].mean-15)+  observed_pred_yo[1].mean * 4)
    M6Raw = torch.tensor(Frame.iloc[observed_pred_yoM6index.item(), 0:6].to_numpy().astype(float)).to(torch.float32)

    #print(observed_pred_yoM,observed_pred_yoM2,observed_pred_yoM3,observed_pred_yoM4)

    observed_pred_yoHC=torch.abs(test_y_actualoHC - observed_pred_yo[0].mean).detach().numpy()
    observed_pred_yoHE= torch.abs(test_y_actualoHE - observed_pred_yo[1].mean).detach().numpy()
    A=100*torch.mean(torch.abs(observed_pred_yoHC/(torch.max(test_y_actualoHC)-torch.min(test_y_actualoHC)))).item()
    B=100*torch.mean(torch.abs(observed_pred_yoHE/(torch.max(test_y_actualoHE)-torch.min(test_y_actualoHE)))).item()
    print("平均预测误差为 推力：",A,"% 效率：",B,"%")




# %%
"""
**Setup the model architecture (k-fold cross-validation)**
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

from training_utils import train, validate, train_MVE, validate_MVE
import torchvision.models as models
from utils import  get_data, get_model

import sys
import datetime
import os
import copy

model_type = sys.argv[1]
data_type = sys.argv[2]

#sys.stdout = open("log_file.txt","a")

current_time = datetime.datetime.now()
 
 # Printing value of now.
print("Current time:", current_time)

print("-"*25)
print("model type:",model_type," | data type: ",data_type)
print("-"*25)

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# number of classes, which is the number of outputs in case of regression
# we train only for one output for regression
num_classes = 1

# mpy.savetxt("foo.csv", a, delimiter=",")learning and training parameters
if (data_type == "CCPP") and model_type == "MVE_NN":
    epochs = 200
if data_type == "Exponential_Function" and (model_type == "MVE_NN" or model_type == "NN"):
    epochs = 300
else:
    epochs = 100

batch_size = 64

if data_type == "Exponential_Function":
    learning_rate = 0.001
else:
    learning_rate = 0.001
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader,valid_loader = get_data(batch_size, data_type)

model = get_model(model_type,data_type,device)

# total parameters and trainable parameters.
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
print("-"*50)

# checking for cuda
print("cuda availability----",torch.cuda.is_available())
print("cuda device count----",torch.cuda.device_count())
print("name of current device----",torch.cuda.get_device_name(0))
print('-'*50)

# counting datapoints
print('total datapoints for training',train_loader.dataset.__len__())
print('total datapoints for testing',valid_loader.dataset.__len__())
# loss function.
mse_criterion = nn.MSELoss()
gauss_nll_criterion = nn.GaussianNLLLoss()

# optimizer
if data_type == "Exponential_Function":
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
else:
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

# %%
train_loss, valid_loss = [], []
variance = []
mse_loss = []
# a default high value to be comnpared later
min_epoch_loss = 10000

for epoch in range(epochs):
    print(f"[info]: epoch {epoch+1} of {epochs}")
    if model_type == "NN":
      train_epoch_loss = train(model,num_classes, train_loader, optimizer, mse_criterion,device)
      valid_epoch_loss,mse_epoch_loss = validate(model,num_classes, valid_loader, mse_criterion,device)
    if model_type == "MVE_NN":
      train_epoch_loss = train_MVE(model,num_classes, train_loader, optimizer, gauss_nll_criterion,device)
      valid_epoch_loss,variance_epoch,mse_epoch_loss = validate_MVE(model,num_classes, valid_loader, mse_criterion,device)
      variance.append(variance_epoch)
      print(f"variance: {variance_epoch:.3f}")
 
    if valid_epoch_loss < min_epoch_loss:
        best_model = copy.deepcopy(model)
        min_epoch_loss = valid_epoch_loss

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    mse_loss.append(mse_epoch_loss)
    print(f"training loss: {train_epoch_loss:.3f}")
    print(f"validation loss: {valid_epoch_loss:.3f}")
    print('-'*50)
# %%
# writting to the risk_assessment file
folder_path =  "/home/nsarna/parameters_model/" + data_type + "/"
filename_model = folder_path + "param_" + model_type + "_split"
os.makedirs(folder_path, exist_ok=True)
torch.save(best_model.state_dict(), filename_model)
# write the training history to a file
np.savetxt("training_history/training_loss_"+model_type+"_split_"+data_type+".csv", mse_loss, delimiter=",")
#sys.stdout.close()
# %%

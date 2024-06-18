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
from utils import get_data, get_model

# 
import sys 
import datetime 
import os
import copy 

model_type = sys.argv[1]
data_type = sys.argv[2]

current_time = datetime.datetime.now()

 # Printing value of now.
print("Current time:", current_time)
# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Learning and training parameters.
num_classes = 1

if data_type == "CCPP" and model_type == "MVE_NN":
    epochs = 200
if data_type == "Exponetial_Function":
    epochs = 300
else:
    epochs = 100
batch_size = 64

if data_type == "Exponential_Function":
    learning_rate = 0.0001
else:
    learning_rate= 0.001 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = get_data(batch_size, data_type)
train_dataset = train_loader.dataset

model = get_model(model_type,data_type,device)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


# checking for CUDA
print("Cuda availability----",torch.cuda.is_available())
print("Cuda device count----",torch.cuda.device_count())
print("name of current device----",torch.cuda.get_device_name(0))
print('-'*50)
# Loss function.
mse_criterion = nn.MSELoss()
gauss_nll_criterion = nn.GaussianNLLLoss()

# %%
from sklearn.model_selection import KFold
k_folds = 10
kfold = KFold(n_splits=k_folds, shuffle=True,random_state = seed)
train_loss_fold = []
min_epoch_error = 10000

for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(
                      train_dataset, 
                      batch_size=batch_size, sampler=train_subsampler)
    valid_loader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=batch_size, sampler=test_subsampler)
  
    breakpoint() 
    if model_type == "NN":
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
    train_loss, valid_loss = [], []
    variance = []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        if model_type == "NN":
            train_epoch_loss = train(model,num_classes,train_loader,optimizer,mse_criterion,device)
            valid_epoch_loss,_ = validate(model,num_classes,valid_loader,mse_criterion,device)

        if model_type == "MVE_NN":
            train_epoch_loss = train_MVE(model,num_classes,train_loader,optimizer,gauss_nll_criterion,device)
            valid_epoch_loss,variance_epoch,_ = validate_MVE(model,num_classes,valid_loader,mse_criterion,device)
            variance.append(variance_epoch)
            print(f"variance epoch: {variance_epoch:.3f}")

        if valid_epoch_loss < min_epoch_error:
            best_model = copy.deepcopy(model)
            min_epoch_error = valid_epoch_loss

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        
        print(f"Training loss: {train_epoch_loss:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}")
    # save the training loss for this fold    
    train_loss_fold.append(train_loss)
    folder_path = "/home/nsarna/parameters_model/" + data_type + "/"
    os.makedirs(folder_path,exist_ok=True)
    file_name = folder_path + "param_" + model_type + "_CV"+str(fold)
    torch.save(best_model.state_dict(),file_name)
print('--------training ended')
# average loss over all the folds
train_loss_fold = np.mean(train_loss_fold,axis=0)
np.savetxt("training_history/training_loss_"+model_type+"_CV_"+data_type+".csv", train_loss_fold, delimiter=",")
#
# %%

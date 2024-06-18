# %%
import torch
import argparse
import numpy as np
import random


#import utils
import os
import sys
import RA_simulation_routines
# %%
# specify model and dataset.
model_type = sys.argv[1]
data_type = sys.argv[2]

# set the number of classes
# this is an old parameter from the classfication problem. Leave at one for regression
num_classes = 1

base_file_model = "/home/nsarna/parameters_model/" + data_type + "/param_" + model_type 
print("INFO: will load model parameters from the folder: " + base_file_model)

# %%
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

score_type = "residual"

data_boundary = {}
data_boundary["tau_type"] = "relative"
# tolerance for the error
if data_type == "CCPP" and model_type == "NN":
    data_boundary["tolerance"] = 0.008
if data_type == "CCPP" and model_type == "MVE_NN":
    data_boundary["tolerance"] = 0.01
if data_type == "Naval_Propulsion" and model_type == "NN":
    data_boundary["tolerance"] = 0.001
if data_type == "Naval_Propulsion" and model_type == "MVE_NN":
    data_boundary["tolerance"] = 0.05
if data_type == "Kin8":
    data_boundary["tolerance"] = 0.15
if data_type == "WineWhite":
    data_boundary["tolerance"] = 0.15
if data_type == "California_Housing":
    data_boundary["tolerance"] = 0.2
if data_type == "Exponential_Function":
    data_boundary["tolerance"] = 0.2



# IID Simulation

# number of calibration points
if data_type == "WineRed":
    n = 500
else:
    n = 1000

RA_simulation_routines.iid_testset_vary(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type)

beta_data = {}
# indexes on which we want to set a value for beta (set depending upon the data to get a 
# reasonable n_effective)
if data_type == "CCPP":
    beta_data["idx_beta"] = [0,-1]
    beta_data["value_beta"] = [-1,6]
if data_type == "California_Housing":
    beta_data["idx_beta"] = [0,-2]
    beta_data["value_beta"] = [-1,1]
if data_type == "Kin8":
    beta_data["idx_beta"] = [0,-1]
    beta_data["value_beta"] = [-1,4]
if data_type == "Naval_Propulsion":
    beta_data["idx_beta"] = [0,-1]
    beta_data["value_beta"] = [-1,6]
if data_type == "WineWhite":
    beta_data["idx_beta"] = [0,-2]
    beta_data["value_beta"] = [-1,1]
if data_type == "Exponential_Function":
    beta_data["idx_beta"] = [0,-1]
    beta_data["value_beta"] = [-1,0.5]


# Simulation with Covariate Shift 
RA_simulation_routines.covariate_shift_testset_vary(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data)


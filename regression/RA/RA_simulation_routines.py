# %%
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import torchvision.models as models
from contextlib import redirect_stdout

import utils
#import utils
from matplotlib.pyplot import figure
from tabulate import tabulate # helps output results in a nice format
#from resnet18 import ResNet, BasicBlock
# netcal packages for binning methods
from netcal import binning
import pandas as pd
import os
import sys
# %%
# run computations for iid data
def iid(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data = {}):

    print("[INFO:] Computing error in misscoverag without covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points

    k_folds = 10
    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader,test_loader = utils.get_calib_test_loader(dataset_remaining,n)

    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader)
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

# %%
    print("-"*20)
    print("Computing calibration scores, output on test points,....")
    
    sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)

    # returns a list of scores for each differen k_fold  calbration sets
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_loader,num_classes)

    print("Finished")
    print("-"*20)
# %%
# %%
    print("-"*20)
    emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)
    if model_type == "NN":
         res_gauss_result = utils.get_coverage_Gaussian(model,test_loader,data_boundary,mean_res_gauss,var_res_gauss,beta)
    if model_type == "MVE_NN":
         res_gauss_result = utils.get_coverage_MVE(model,test_loader,data_boundary,beta)
  
    cp_result_CV = utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,test_loader,w_calib_CV,beta,data_boundary,k_folds)
    cp_split =  utils.get_CP_coverage([sorted_scores_on_calib],[model],model,test_loader,[w_calib],beta,data_boundary,1)

    print("cp split coverage",cp_split)
    print("empirical coverage",emp_result)
    print("res gauss coverage",res_gauss_result)
    print('cp CV coverage',cp_result_CV)

    methods = ["CP-S",'CP-CV','Res-Gauss',"emp coverage","tolerance"]
    values = [cp_split.item(),cp_result_CV.item(),res_gauss_result,emp_result.item(),data_boundary["tolerance"]]
    print('-'*50)
    return values,methods


# same as above but we average over the different calibration and test sets
def iid_average(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data = {}):

    print("[INFO:] Computing error in misscoverag without covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points

    k_folds = 10
    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader_list,test_loader_list = utils.get_calib_test_loader_splitted(dataset_remaining,n,seed) 
    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader_list[0])
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

# %%
    print("-"*20)
    print("Computing calibration scores, output on test points,....")
 
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    values = []

    for i in tqdm(range(len(calib_loader_list))):
      calib_loader = calib_loader_list[i]
      test_loader = test_loader_list[i]
      sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)

    # returns a list of scores for each differen k_fold  calbration sets
      mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_loader,num_classes)

      
      emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)
      if model_type == "NN":
          res_gauss_result = utils.get_coverage_Gaussian(model,test_loader,data_boundary,mean_res_gauss,var_res_gauss,beta)
      if model_type == "MVE_NN":
          res_gauss_result = utils.get_coverage_MVE(model,test_loader,data_boundary,beta)
  
      cp_result_CV = utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,test_loader,w_calib_CV,beta,data_boundary,k_folds)
      cp_split =  utils.get_CP_coverage([sorted_scores_on_calib],[model],model,test_loader,[w_calib],beta,data_boundary,1)

      values.append([cp_split.item(),cp_result_CV.item(),res_gauss_result,emp_result.item(),data_boundary["tolerance"]])

    methods = ["CP-S",'CP-CV','Res-Gauss',"emp coverage","tolerance"]
 
    return values,methods


# same as iid but we study the variation of miss-coverage under test set variation
def iid_testset_vary(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data = {}):

    print("[INFO:] Computing error in misscoverag without covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points

    k_folds = 10
    k_folds_test = 20

    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader,test_loader = utils.get_calib_test_loader(dataset_remaining,n)
    test_loader_splitted = utils.get_splitted_loader(test_loader,k_folds_test,seed)
    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader)
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

# %%
    print("-"*20)
    print("Computing calibration scores, output on test points,....")
    
    sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)

    # returns a list of scores for each differen k_fold  calbration sets
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_loader,num_classes)

    print("Finished")
    print("-"*20)
# %%
# %%
    print("-"*20)
    emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)


    cp_result_CV = []
    cp_split = []
    res_gauss_result = []

    for loader in test_loader_splitted:
         cp_result_CV.append(utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,loader,w_calib_CV,beta,data_boundary,k_folds))
         cp_split.append(utils.get_CP_coverage([sorted_scores_on_calib],[model],model,loader,[w_calib],beta,data_boundary,1))
         if model_type == "NN":
             res_gauss_result.append(utils.get_coverage_Gaussian(model,loader,data_boundary,mean_res_gauss,var_res_gauss,beta))
         if model_type == "MVE_NN":
             res_gauss_result.append(utils.get_coverage_MVE(model,loader,data_boundary,beta))
 
    for i in range(k_folds_test):
       methods = ["CP-S",'CP-CV','Res-Gauss',"emp coverage","tolerance"]
       values = [cp_split[i].item(),cp_result_CV[i].item(),res_gauss_result[i],emp_result.item(),data_boundary["tolerance"]]
       utils.output_error_table(methods,values,"without" + "_" + str(i),data_type,model_type)
    print("-"*50)
    print("Finished")


# routines for with covariate shift
def covariate_shift(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data):

    print("[INFO:] Computing error in misscoverag with covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points
    k_folds = 10
    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader,test_loader = utils.get_calib_test_loader(dataset_remaining,n)


    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader)
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

    # %%
    print("-"*20)
    print("Computing calibration scores, output on test points,....")
    # beta for exponential weighting, expression exp(xT beta)
    score_type = "residual"
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    
    for count, i in enumerate(beta_data["idx_beta"]):
        beta[i] = beta_data["value_beta"][count]

    sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)
    n_effective = torch.square(torch.sum(w_calib)).item()/torch.sum(torch.square(w_calib)).item()
    print('effective calibration points: ',n_effective)
    # extract from the current calib loader n_effective points to form the calib_effective_loader
    calib_effective_loader = utils.get_calib_effective(calib_loader,n_effective,n)
    # compute the scores and the weights on the reduced calibration set
    sorted_scores_on_calib_effective,w_calib_effective = utils.get_scores(model,calib_effective_loader,score_type,num_classes,beta)
    

# returns a list of scores for each differen k_fold  calbration sets
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_effective_loader,num_classes)

    print("Finished")
    print("-"*20)


    emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)
    if model_type == "NN":
         res_gauss_result = utils.get_coverage_Gaussian(model,test_loader,data_boundary,mean_res_gauss,var_res_gauss,beta)
    if model_type == "MVE_NN":
         res_gauss_result = utils.get_coverage_MVE(model,test_loader,data_boundary,beta)
    
    cp_result_CV = utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,test_loader,w_calib_CV,beta,data_boundary,k_folds)
    # run the split routine. The full model and the model for CP both are the same. 
    cp_split =  utils.get_CP_coverage([sorted_scores_on_calib],[model],model,test_loader,[w_calib],beta,data_boundary,1)
    # the testing distribution could be different but no weights are added 
    cp_split_unweighted =  utils.get_CP_coverage_unweighted([sorted_scores_on_calib_effective],[model],model,test_loader,beta,data_boundary,1)

    print("cp split coverage",cp_split)
    print("empirical coverage",emp_result)
    print("res gauss coverage",res_gauss_result)
    print('cp CV coverage',cp_result_CV)
    print("CP split unweighted",cp_split_unweighted)

    filename_beta_vector = "results/beta_" + data_type + "_" + model_type + "_" +  "with" + ".txt"

    np.savetxt(filename_beta_vector,beta.numpy())

    methods = ['CP-SW','CP-CVW',"CP-S",'Res-Gauss',"emp coverage","tolerance","n_effective"]
    values = [cp_split.item(),cp_result_CV.item(),cp_split_unweighted.item(),res_gauss_result,emp_result.item(),data_boundary["tolerance"],n_effective]
    return values,methods

# same as above but for multiple runs
def covariate_shift_average(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data):

    print("[INFO:] Computing error in misscoverag with covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points
    k_folds = 10
    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader_list,test_loader_list = utils.get_calib_test_loader_splitted(dataset_remaining,n,seed)

    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader_list[0])
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

    # %%
    # beta for exponential weighting, expression exp(xT beta)
    score_type = "residual"
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    
    for count, i in enumerate(beta_data["idx_beta"]):
        beta[i] = beta_data["value_beta"][count]

    # returns a list of scores for each differen k_fold  calbration sets
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    values = []

    for i in tqdm(range(len(calib_loader_list))):
        calib_loader = calib_loader_list[i]
        test_loader = test_loader_list[i]
        sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)
        n_effective = torch.square(torch.sum(w_calib)).item()/torch.sum(torch.square(w_calib)).item()
    
        # extract from the current calib loader n_effective points to form the calib_effective_loader
        calib_effective_loader = utils.get_calib_effective(calib_loader,n_effective,n)
         # compute the scores and the weights on the reduced calibration set
        sorted_scores_on_calib_effective,w_calib_effective = utils.get_scores(model,calib_effective_loader,score_type,num_classes,beta)
    

        mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_effective_loader,num_classes)


        emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)
        if model_type == "NN":
             res_gauss_result = utils.get_coverage_Gaussian(model,test_loader,data_boundary,mean_res_gauss,var_res_gauss,beta)
        if model_type == "MVE_NN":
             res_gauss_result = utils.get_coverage_MVE(model,test_loader,data_boundary,beta)
    
        cp_result_CV = utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,test_loader,w_calib_CV,beta,data_boundary,k_folds)
        # run the split routine. The full model and the model for CP both are the same. 
        cp_split =  utils.get_CP_coverage([sorted_scores_on_calib],[model],model,test_loader,[w_calib],beta,data_boundary,1)
        # testing distribution is different but not weights added
        cp_split_unweighted =  utils.get_CP_coverage_unweighted([sorted_scores_on_calib_effective],[model],model,test_loader,beta,data_boundary,1)
        values.append([cp_split.item(),cp_result_CV.item(),cp_split_unweighted.item(),res_gauss_result,emp_result.item(),data_boundary["tolerance"],n_effective])
 


    methods = ['CP-SW','CP-CVW',"CP-S",'Res-Gauss',"emp coverage","tolerance","n_effective"]
    return values,methods


def covariate_shift_testset_vary(model_type,data_type,num_classes,data_boundary,n,base_file_model,score_type,beta_data):

    print("[INFO:] Computing error in misscoverag with covariate shift")
    # %%
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    # number of calibration points
    k_folds = 10
    k_folds_test = 20
    training_dataset,dataset_remaining = utils.get_data(data_type)
    calib_loader,test_loader = utils.get_calib_test_loader(dataset_remaining,n)
    test_loader_splitted = utils.get_splitted_loader(test_loader,k_folds_test,seed)
    # returns a list of dataloaders that iterate over the validation set of CV
    calib_loader_CV = utils.get_calib_CV(training_dataset,seed,k_folds)

    n_features =utils.get_n_features(calib_loader)
    print('total features:',n_features)
    # %%
    #all computations on the cpu
    device = torch.device("cpu")
    # model trained on the entire training dataset (unlike CV)
    model =utils.get_model(model_type,data_type,device,base_file_model + "_split")
    # returns a list of models for the 
    models_CV = utils.get_model_CV(model_type,data_type,device,base_file_model,k_folds)

    # %%
    print("-"*20)
    print("Computing calibration scores, output on test points,....")
    # beta for exponential weighting, expression exp(xT beta)
    score_type = "residual"
    beta = torch.tensor([0]*n_features) # same as in Tibshirani's paper
    
    for count, i in enumerate(beta_data["idx_beta"]):
        beta[i] = beta_data["value_beta"][count]

    sorted_scores_on_calib,w_calib = utils.get_scores(model,calib_loader,score_type,num_classes,beta)
    n_effective = torch.square(torch.sum(w_calib)).item()/torch.sum(torch.square(w_calib)).item()
    print('effective calibration points: ',n_effective)
    # extract from the current calib loader n_effective points to form the calib_effective_loader
    calib_effective_loader = utils.get_calib_effective(calib_loader,n_effective,n)
    # compute the scores and the weights on the reduced calibration set
    sorted_scores_on_calib_effective,w_calib_effective = utils.get_scores(model,calib_effective_loader,score_type,num_classes,beta)
    

# returns a list of scores for each differen k_fold  calbration sets
    sorted_scores_on_calib_CV, w_calib_CV = utils.get_scores_CV(models_CV,calib_loader_CV,score_type,num_classes,beta)
    mean_res_gauss, var_res_gauss = utils.get_error_mean_variance(model,calib_effective_loader,num_classes)

    print("Finished")
    print("-"*20)


    emp_result = utils.get_emp_coverage(model,test_loader,data_boundary,beta)
  
    cp_result_CV = []
    cp_split = []
    res_gauss_result = []

    for loader in test_loader_splitted:
          cp_result_CV.append(utils.get_CP_coverage(sorted_scores_on_calib_CV,models_CV,model,loader,w_calib_CV,beta,data_boundary,k_folds))
         # run the split routine. The full model and the model for CP both are the same. 
          cp_split.append(utils.get_CP_coverage([sorted_scores_on_calib],[model],model,loader,[w_calib],beta,data_boundary,1))
         # the testing distribution could be different but no weights are added 
          if model_type == "NN":
               res_gauss_result.append(utils.get_coverage_Gaussian(model,loader,data_boundary,mean_res_gauss,var_res_gauss,beta))
          if model_type == "MVE_NN":
               res_gauss_result.append(utils.get_coverage_MVE(model,loader,data_boundary,beta))
 
    cp_split_unweighted =  utils.get_CP_coverage_unweighted([sorted_scores_on_calib_effective],[model],model,test_loader,beta,data_boundary,1)

    for i in range(k_folds_test):
         methods = ['CP-SW','CP-CVW',"CP-S",'Res-Gauss',"emp coverage","tolerance","n_effective"]
         values = [cp_split[i].item(),cp_result_CV[i].item(),cp_split_unweighted.item(),res_gauss_result[i],emp_result.item(),data_boundary["tolerance"],n_effective]
         shift = "with"+ "_" + str(i)
         utils.output_error_table(methods,values,shift,data_type,model_type)
    print("-"*50)
    print('Finished')


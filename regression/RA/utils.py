import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torchvision.models as models
import importlib
import sys
from statistics import NormalDist
import math
import numpy as np
import pandas as pd
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../training_regression/')
import custom_datasets
import NN_arch 
from sklearn.model_selection import KFold

importlib.reload(NN_arch)
importlib.reload(custom_datasets)
plt.style.use('ggplot')

def get_data(data_type= "California_Housing"):

    read_data = False
    if data_type == "California_Housing":
        dataset = custom_datasets.California_Housing()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.7,0.3])


    if data_type == "Kin8":
        dataset = custom_datasets.Kin8()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.7,0.3])


    if data_type == "Naval_Propulsion":
        dataset = custom_datasets.NavalPropulsion()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.7,0.3])


    if data_type == "CCPP":
        dataset = custom_datasets.CCPP()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.7,0.3])


    if data_type == "WineRed":
        dataset = custom_datasets.WineRed()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.3,0.7])

    if data_type == "WineWhite":
        dataset = custom_datasets.WineWhite()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.3,0.7])

    if data_type == "Exponential_Function":
        dataset = custom_datasets.Exponential_Function()
        read_data = True
        dataset_train,dataset_remaining = random_split(dataset,[0.2,0.8])


    return dataset_train, dataset_remaining

# n is the number of calibration points
def get_calib_test_loader(dataset_remaining,n=1000):

    # split the data not used for training into calibration and testing
    calib_set,test_set = random_split(dataset_remaining, [n, dataset_remaining.__len__() - n])
    # keep number of batches to one
    calib_loader = DataLoader(calib_set,batch_size = calib_set.__len__(),shuffle = True)
    test_loader = DataLoader(test_set,batch_size = test_set.__len__(),shuffle = True)


    return calib_loader, test_loader

# n is the number of calibration points
def get_calib_test_loader_splitted(dataset_remaining,n=1000,seed = 42):
    
    calib_loader_list = []
    test_loader_list = []
    n_trials = 100
    ids = [i for i in range(len(dataset_remaining))]

    for i in range(n_trials):
   
      ids = [i for i in range(len(dataset_remaining))]
      np.random.shuffle(ids)
      calib_ids = ids[:n]
      test_ids = ids[n:]
    
      # Sample elements randomly from a given list of ids, no replacement.
      test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
      # Define data loaders for training and testing data in this fold
      test_loader_list.append(torch.utils.data.DataLoader(
                      dataset_remaining,
                      batch_size=len(test_ids), sampler=test_subsampler))
      # Sample elements randomly from a given list of ids, no replacement.
      calib_subsampler = torch.utils.data.SubsetRandomSampler(calib_ids)
      
      # Define data loaders for training and testing data in this fold
      calib_loader_list.append(torch.utils.data.DataLoader(
                      dataset_remaining,
                      batch_size=len(calib_ids), sampler=calib_subsampler))

      
    return calib_loader_list,test_loader_list


# splits a loader into k0folds and returns the resulting loaders. Used for studying the 
# variaton of miss-coverage over testsets
def get_splitted_loader(data_loader,kfolds,seed):
   
    kfold = KFold(n_splits = kfolds,shuffle=True,random_state = seed)
    valid_loader = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_loader.dataset)):
    
      # Sample elements randomly from a given list of ids, no replacement.
      test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
      # Define data loaders for training and testing data in this fold
      valid_loader.append(torch.utils.data.DataLoader(
                      data_loader.dataset,
                      batch_size=len(test_ids), sampler=test_subsampler))

    return valid_loader
 

# reduce the size of the calibration set for the covariate shift problem
def get_calib_effective(calib_loader,n_effective,n_calib):


    if n_effective > n_calib:
        raise Exception("n_effective cannot be larger than the n_calib.")


    if n_effective == n_calib:
        return calib_loader

    # split the current loader over the calibration set into two parts. First part goes into the effective 
    # calibration loader. 
    calib_effective,_ = random_split(calib_loader.dataset,[n_effective/n_calib,1-n_effective/n_calib])

    return DataLoader(calib_effective,batch_size = calib_effective.__len__(),shuffle = True)

# every CV iteration has a different calibration set. We collect all these calibration sets in the 
# routine below. 
def get_calib_CV(train_dataset,seed,k_folds = 10):
    kfold = KFold(n_splits=k_folds, shuffle=True,random_state = seed)
    result = []


    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
          # Sample elements randomly from a given list of ids, no replacement.
        valid_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        # Define data loaders for training and testing data in this fold
        valid_loader = torch.utils.data.DataLoader(
                      train_dataset,
                      batch_size=len(test_ids), sampler=valid_subsampler)
        result.append(valid_loader)

    return result
 
   
def get_model(model_type,data_type,device,filename_model_parameters):

    model_read = False

    if model_type == "NN" and (data_type == "California_Housing" or data_type == "Kin8"):
        model = NN_arch.NN_input8_output1()
        model_read = True

    if model_type == "NN" and (data_type == "Naval_Propulsion"):
        model = NN_arch.NN_input16_output1()
        model_read = True

    if model_type == "NN" and data_type == "CCPP":
        model = NN_arch.NN_input4_output1()
        model_read = True

    if model_type == "NN" and (data_type == "WineRed" or data_type == "WineWhite"):
        model = NN_arch.NN_input11_output1()
        model_read = True


    if model_type == "NN" and data_type == "Exponential_Function":
        model = NN_arch.NN_input2_output1()
        model_read = True

    if model_type == "MVE_NN" and data_type == "Exponential_Function":
        model = NN_arch.NN_input2_output2()
        model_read = True

    if model_type == "MVE_NN" and (data_type == "California_Housing" or data_type == "Kin8"):
        model = NN_arch.NN_input8_output2()
        model_read = True

    if model_type == "MVE_NN" and (data_type == "Naval_Propulsion"):
#        model = NN_arch.NN_input16_output2()
        model = NN_arch.NN_input16_output2_deeper()
        model_read = True

    if model_type == "MVE_NN" and data_type == "CCPP":
        model = NN_arch.NN_input4_output2()
        model_read = True

    if model_type == "MVE_NN" and (data_type == "WineRed" or data_type == "WineWhite"):
        model = NN_arch.NN_input11_output2()
        model_read = True

    if model_type == "MVE_NN" and data_type == "Exponential_Function":
        model = NN_arch.NN_input2_output2()
        model_read = True



    if not model_read:
        raise Expection("didnt read the model")
    # load pre-trained model parameters
    model.load_state_dict(torch.load(filename_model_parameters))
    model = model.to(torch.device("cpu"))
   
    return model
  
# returns a list of models developed during CV
def get_model_CV(model_type,data_type,device,filename_model,k_folds):
    result = []
    for fold in range(k_folds):
        filename_this_fold = filename_model + "_CV" + str(fold)
        result.append(get_model(model_type,data_type,device,filename_this_fold))

    return result


def get_model_output(model,data_loader,num_classes):
    model.eval()
    
    with torch.no_grad():
        # loop over the data loader 
        for data in data_loader:
            features = data["features"]
            labels = data["target"]
            # add the direction of the batch
            
            prediction = model(features)
            
            return prediction,labels,features          

# CP scores based on residuals
def get_scores(model,calib_loader,score_type,num_classes,beta):
        
    model_output,true_label,features = get_model_output(model,calib_loader,num_classes)

    if score_type == "residual":
        scores =torch.abs(model_output[:,0] - true_label)
    if score_type == "residual_variance":
        scores = torch.div(torch.abs(model_output[:,0]-true_label),torch.abs(model_output[:,1]))

    w_calib = get_w(features,beta)
    # sort the scores, useful for later
    scores_sorted,index = torch.sort(scores)

    return scores_sorted,w_calib[index]

# CP scores for the CV fold model
def get_scores_CV(models,calib_loaders,score_type,num_classes,beta):
    kfolds = len(calib_loaders)
    scores = []
    w_calib = []
    for fold in range(kfolds):
        scores_this_fold, w_calib_this_fold = get_scores(models[fold],calib_loaders[fold],score_type,num_classes,beta)
        scores.append(scores_this_fold)
        w_calib.append(w_calib_this_fold)

    return scores, w_calib

# mean and variance of the error for the residual method
def get_error_mean_variance(model,calib_loader,num_classes):
         
    model_output,true_label,features = get_model_output(model,calib_loader,num_classes)
    error_values = torch.abs(model_output[:,0]-true_label)

    return torch.mean(error_values),torch.var(error_values)
   

# get upper and lower boundary on the test points
# output = only the mean prediction without any variance
def get_boundary(output,data_boundary):

    if data_boundary["tau_type"] == "relative":
        return output * (1-data_boundary["tolerance"]), output * (1 +data_boundary["tolerance"])
    if data_boundary["tau_type"] == "absolute":
        return output - data_boundary["tolerance"], output + data_boundary["tolerance"]


def get_w(X,beta):
    # sum along the features. Implements exp(x^T beta) weights
    return torch.exp(torch.sum(X * beta,axis=1)) 

# Fit Gaussian over the residuals and compute the coverage
def get_coverage_Gaussian(model,test_loader,data_boundary,mean_residual_Gaussian,var_residual_Gaussian,beta):
    for data in test_loader:
        features = data["features"]
        labels = data["target"]
        n_test = len(labels) # number of test points
 
        # take care of the shift via re-sampling
        w_test = get_w(features,beta)
        # take care of covariate shift by resampling
        labels,features,w_test = resample_with_covariate(labels,features,w_test)
            
        model_output = model(features)[:,0]
        # add Gaussian distribution over the residual to our mena predictor
        mean_output = model_output + mean_residual_Gaussian
        sigma_output = math.sqrt(var_residual_Gaussian)
        # lower boundary of the interval I(X)
        boundary_low,boundary_up = get_boundary(model_output,data_boundary)

    # convert to numpy arrays (for the Gaussian function library)
    mean_output = mean_output.cpu().detach().numpy()
    boundary_low = boundary_low.cpu().detach().numpy()
    boundary_up = boundary_up.cpu().detach().numpy()

    coverage = 0
    # loop over the test poinits
    for i in range(len(mean_output)):
        normal_dist = NormalDist(mu = mean_output[i],sigma = sigma_output)
        cdf_up = normal_dist.cdf(boundary_up[i])
        cdf_low = normal_dist.cdf(boundary_low[i])
        coverage += (cdf_up - cdf_low)

    return coverage/len(mean_output)

# get coverage with the MVE architecture
def get_coverage_MVE(model,test_loader,data_boundary,beta):
    for data in test_loader:
        features = data["features"]
        labels = data["target"]
        n_test = len(labels) # number of test points
 
        # take care of the shift via re-sampling
        w_test = get_w(features,beta)
        # take care of covariate shift by resampling
        labels,features,w_test = resample_with_covariate(labels,features,w_test)
            
        model_output = model(features)
        # add Gaussian distribution over the residual to our mena predictor
        mean_output = model_output[:,0]
        # standard deviation outputed from the model
        sigma_output = torch.abs(model_output[:,1])
        # lower boundary of the interval I(X)
        boundary_low,boundary_up = get_boundary(model_output[:,0],data_boundary)

    # convert to numpy arrays (for the Gaussian function library)
    mean_output = mean_output.cpu().detach().numpy()
    sigma_output = sigma_output.cpu().detach().numpy()
    boundary_low = boundary_low.cpu().detach().numpy()
    boundary_up = boundary_up.cpu().detach().numpy()

    coverage = 0
    # loop over the test poinits
    for i in range(len(mean_output)):
        normal_dist = NormalDist(mu = mean_output[i],sigma = sigma_output[i])
        cdf_up = normal_dist.cdf(boundary_up[i])
        cdf_low = normal_dist.cdf(boundary_low[i])
        coverage += (cdf_up - cdf_low)

    return coverage/len(mean_output)


# resample for covariate shift
# w = weights on the test points
def resample_with_covariate(labels,features,w):
    
    normalized_w = w/torch.max(w)
    n_test = len(w) ## n : number of test points
    
    indices = [] ## indices : vector containing indices of the sampled data
    
    while(len(indices) < n_test): ## Draw samples until have sampled the same number of test points as earlier
        proposed_indices = torch.rand(n_test) <= normalized_w
        proposed_indices = proposed_indices.nonzero().squeeze(1)

        # initialize for the first iteration
        if len(indices) == 0:
            indices = proposed_indices
        else:
            indices = torch.cat((indices,proposed_indices))

    # selecting the top n_test points (length of indices as set above might be higher)
    indices = indices[:n_test]
    return labels[indices],features[indices,:],w[indices]


# works for both split and CV
# set kfolds to one for split CP
def get_CP_coverage(sorted_scores,model,model_full,test_loader,w_calib,beta,data_boundary,kfolds):

    # sum over all the w computed on the calibration points
    sum_w_calib =sum([torch.sum(i) for i in w_calib])
    
    for data in test_loader:
        features = data["features"]
        labels = data["target"]
        # resampling to take into account covariate shift. If no covariate shift then automatically 
        # returns the correct samples
        n_test = len(labels)
        w_test = get_w(features,beta)
        # take care of covariate shift by resampling
        labels,features,w_test = resample_with_covariate(labels,features,w_test)
        # model trained on the entire training dataset
        model_output_full = model_full(features)
        mean_prediction = model_output_full[:,0]
        boundary_low,boundary_up = get_boundary(mean_prediction,data_boundary)
        
        coverage_up = torch.tensor([0]*n_test)
        coverage_low = torch.tensor([0] * n_test)

        for fold in range(kfolds):
            model_output = model[fold](features)

             # denominator of weights on the test points, normalization factor
             # sum_w is list that has number of testpoint number of elements
            sum_w = sum_w_calib + w_test
             # divide the weights on the calibration points by the normalizing factor to get the weights.
            # the [:,None] referecing is for matrix broadcasting
            # p_wegiths is a matrix of size (num test points,num of calibration points)
            p_weights = w_calib[fold].repeat(n_test,1)/sum_w[:,None]
       
            # Vi_plus is  a matrix of size (n_test points,number of calibration points)
            Vi_plus = model_output[:,0].unsqueeze(1) + sorted_scores[fold].repeat(n_test,1)
            Vi_minus = model_output[:,0].unsqueeze(1) - sorted_scores[fold].repeat(n_test,1)
        
            # Create a mask for points that are inside the upper boundary. Unsqueeze for mkaing the 
            # array broadcastable by adding an additional dimension to it.
            points_in_up = torch.le(Vi_plus,boundary_up.unsqueeze(1))
            # alpha corresponding to the upper boundary

            # sum up weights of all the calibration points that are inside the boundary. 
            coverage_up =torch.sum(p_weights * points_in_up,axis=1) + coverage_up
            # same as above for the lower boundary
            points_in_low = torch.ge(Vi_minus,boundary_low.unsqueeze(1))
            # sum up all the weights corresponding to those calibration points that are inside the 
            # boundary
            coverage_low = torch.sum(points_in_low *  p_weights,axis = 1) + coverage_low

    return torch.mean(torch.minimum(coverage_up,coverage_low))

# the test set might result from a different distribution but we dont weight the CP technique
def get_CP_coverage_unweighted(sorted_scores,model,model_full,test_loader,beta,data_boundary,kfolds):

    w_calib = [torch.ones(sorted_scores[0].shape)]
    # sum over all the w computed on the calibration points
    sum_w_calib =sum([torch.sum(i) for i in w_calib])
    
    for data in test_loader:
        features = data["features"]
        labels = data["target"]
        # resampling to take into account covariate shift. If no covariate shift then automatically 
        # returns the correct samples
        n_test = len(labels)
        w_test = get_w(features,beta)
        # take care of covariate shift by resampling
        labels,features,_ = resample_with_covariate(labels,features,w_test)
        # rest the weights to one on the test points (no weighting applied)
        w_test = torch.ones(w_test.shape)
        # model trained on the entire training dataset
        model_output_full = model_full(features)
        boundary_low,boundary_up = get_boundary(model_output_full[:,0],data_boundary)
        
        coverage_up = torch.tensor([0]*n_test)
        coverage_low = torch.tensor([0] * n_test)

        for fold in range(kfolds):
            model_output = model[fold](features)

             # denominator of weights on the test points, normalization factor
             # sum_w is list that has number of testpoint number of elements
            sum_w = sum_w_calib + w_test
             # divide the weights on the calibration points by the normalizing factor to get the weights.
            # the [:,None] referecing is for matrix broadcasting
            # p_wegiths is a matrix of size (num test points,num of calibration points)
            p_weights = w_calib[fold].repeat(n_test,1)/sum_w[:,None]
       
            # Vi_plus is  a matrix of size (n_test points,number of calibration points)
            Vi_plus = model_output[:,0].unsqueeze(1) + sorted_scores[fold].repeat(n_test,1)
            Vi_minus = model_output[:,0].unsqueeze(1) - sorted_scores[fold].repeat(n_test,1)
        
            # Create a mask for points that are inside the upper boundary. Unsqueeze for mkaing the 
            # array broadcastable by adding an additional dimension to it.
            points_in_up = torch.le(Vi_plus,boundary_up.unsqueeze(1))
            # alpha corresponding to the upper boundary

            # sum up weights of all the calibration points that are inside the boundary. 
            coverage_up =torch.sum(p_weights * points_in_up,axis=1) + coverage_up
            # same as above for the lower boundary
            points_in_low = torch.ge(Vi_minus,boundary_low.unsqueeze(1))
            # sum up all the weights corresponding to those calibration points that are inside the 
            # boundary
            coverage_low = torch.sum(points_in_low *  p_weights,axis = 1) + coverage_low

    return torch.mean(torch.minimum(coverage_up,coverage_low))


# get empirical coverage by counting
def get_emp_coverage(model,test_loader,data_boundary,beta):
    model.eval()

    with torch.no_grad():
      for data in test_loader:
            features = data["features"]
            labels = data["target"]
            w = get_w(features,beta)

            labels,features,_ = resample_with_covariate(labels,features,w)
            model_output = model(features)
            mean_prediction = model_output[:,0]
            boundary_low,boundary_up = get_boundary(mean_prediction,data_boundary)

            labels_in = torch.logical_and(torch.le(boundary_low,labels),torch.le(labels,boundary_up)).float()
            emp_coverage = torch.mean(labels_in)
   
    return emp_coverage
    

 
       
# Output a table containing all the results
def output_error_table(methods,values,shift,data_type,model_type):

     
     df = pd.DataFrame(data = values,columns = methods)
     

     filename_error_table = "results/error_table_" + data_type + "_" + model_type + "_" +  shift + ".csv" 

     isExist = os.path.exists("results")
     if not isExist:
           # Create a new directory because it does not exist
              os.makedirs("results")

     # write the error table to excel
     df.to_csv(filename_error_table,index = False)

def write_results(values,methods,filename):

     df = pd.DataFrame(data = values,columns = methods)
     
     isExist = os.path.exists("results")
     if not isExist:
           # Create a new directory because it does not exist
              os.makedirs("results")

     # write the error table to excel
     df.to_csv(filename,index = False)


def get_n_features(data_loader):
    for data in data_loader:
        features = data["features"]
        return features.shape[1]


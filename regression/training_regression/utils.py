import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.nn as nn
import torchvision.models as models
import NN_arch 
import importlib
import custom_datasets

importlib.reload(NN_arch)
importlib.reload(custom_datasets)
plt.style.use('ggplot')


def get_data(batch_size=64,data_type= "California_Housing"):

    read_data = False
    if data_type == "California_Housing":
        dataset = custom_datasets.California_Housing()
        read_data = True

    if data_type == "Swiss_Roll":
        dataset = custom_datasets.Swiss_Roll()
        read_data = True

    if data_type == "Exponential_Function":
        dataset = custom_datasets.Exponential_Function()
        read_data = True

    if data_type == "Kin8":
        dataset = custom_datasets.Kin8()
        read_data = True


    if data_type == "Naval_Propulsion":
        dataset = custom_datasets.NavalPropulsion()
        read_data = True

    if data_type == "CCPP":
        dataset = custom_datasets.CCPP()
        read_data = True

    if data_type == "WineRed":
        dataset = custom_datasets.WineRed()
        read_data = True

    if data_type == "WineWhite":
        dataset = custom_datasets.WineWhite()
        read_data = True


    if data_type == "WineRed" or data_type == "WineWhite":
        dataset_train,dataset_valid = random_split(dataset,[0.3,0.7])
    if data_type == "Swiss_Roll" or data_type == "Exponential_Function":
        dataset_train,dataset_valid = random_split(dataset,[0.2,0.8])
    else:
        dataset_train,dataset_valid = random_split(dataset,[0.7,0.3])

    if not read_data:
        error("Didn't read data. Check the data type.")
    # Create data loaders.
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, valid_loader

   
def get_model(model_type,data_type,device = "cpu"):


    if model_type == "NN" and (data_type == "Swiss_Roll" or data_type == "Exponential_Function") :
        model = NN_arch.NN_input2_output1()

    if model_type == "MVE_NN" and (data_type == "Swiss_Roll" or data_type == "Exponential_Function"):
        model = NN_arch.NN_input2_output2()

    if model_type == "NN" and (data_type == "California_Housing" or data_type == "Kin8"):
        model = NN_arch.NN_input8_output1()

    if model_type == "MVE_NN" and (data_type == "California_Housing" or data_type == "Kin8"):
        model = NN_arch.NN_input8_output2()

    if model_type == "NN" and (data_type == "Naval_Propulsion"):
        model = NN_arch.NN_input16_output1()

    if model_type == "MVE_NN" and (data_type == "Naval_Propulsion"):
     #   model = NN_arch.NN_input16_output2()
         model = NN_arch.NN_input16_output2_deeper()


    if model_type == "NN" and data_type == "CCPP":
        model = NN_arch.NN_input4_output1()

    if model_type == "MVE_NN" and data_type == "CCPP":
        model = NN_arch.NN_input4_output2()


    if model_type == "NN" and (data_type == "WineRed" or data_type == "WineWhite"):
        model = NN_arch.NN_input11_output1()


    if model_type == "MVE_NN" and (data_type == "WineRed" or data_type == "WineWhite"):
        model = NN_arch.NN_input11_output2()
 
    model = model.to(device)
    return model
    

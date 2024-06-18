# %%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import custom_datasets
from utils import get_data
import pandas as pd

df = pd.read_csv("data/test_loan_prediction_clean.csv")
print(df.shape)
print(df.head())
print(df['Loan_Status'].iloc[0])

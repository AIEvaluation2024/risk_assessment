import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from sklearn.datasets import fetch_california_housing,make_swiss_roll

# routines for standardisation of data. Remain the same for all the datasets defined later
class helper_routines():
    def standardize_data(self,df):
        for i in range(df.shape[1]):
            # standard deviation for the i-the feature
            s_dev = np.std(df[:,i])
            # normalize incase you find a zero
            if s_dev == 0:
                s_dev = 0.001
            df[:,i] = (df[:,i]-np.mean(df[:,i]))/s_dev

        return df


    def return_label_feature(self,idx,label,features):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get the target and the features corresponding to the index
        target = label[idx]
        features = features[idx,:]

        sample = {"features":features,"target":target}
        # weights from pytorch are in float32 format 
        sample["features"] = torch.from_numpy(np.asarray(features))
        sample["target"] = torch.from_numpy(np.asarray(target))

        # change to float32
        sample["features"] = sample["features"].to(torch.float)
        sample["target"] = sample["target"].to(torch.float)

        return sample

# generate simulated dataset

def generate_simulated_data(n_samples): 

    X1 = np.random.normal(loc = 0,scale = 1.0,size =int( np.sqrt(n_samples)))
    X2 = np.random.normal(loc = 0,scale = 1.0,size = int(np.sqrt(n_samples)))
    beta1 = 0.3
    beta2 = 0.8
    X = []
    Y = []

    for x1 in X1:
        for x2 in X2:
            X.append([x1,x2])
            noise = np.random.normal(loc = 0, scale = 0.1)
            y_value = np.exp(-x1 * x2) * beta1 + np.exp(-x2 * x2) * beta2 + noise  
            Y.append(y_value)
    Y = np.array(Y)
    X = np.array(X)

    return X,Y

# swiss roll simulated dataset
class Swiss_Roll(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        X = make_swiss_roll(n_samples = 10000,noise = 0.01)[0]
        self.features = X[:,:2]
        self.label = X[:,2]
        self.transform = transform

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
        return self.helper_routines.return_label_feature(idx,self.label,self.features)

class Exponential_Function(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        self.features,self.label = generate_simulated_data(n_samples = 10000)
        self.transform = transform
        self.helper_routines = helper_routines()
        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)

    def __getitem__(self,idx):
        return self.helper_routines.return_label_feature(idx,self.label,self.features)

class California_Housing(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        self.features,self.label = fetch_california_housing(return_X_y = True)
        self.transform = transform

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
        return self.helper_routines.return_label_feature(idx,self.label,self.features)


# class for the kin8 dataset
class Kin8(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        df = pd.read_csv('/home/nsarna/data/dataset_2175_kin8nm.csv')

        self.label = df["y"].to_numpy()
        df = df.drop(["y"],axis=1)
        self.features = df.to_numpy()

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
        return self.helper_routines.return_label_feature(idx,self.label,self.features)

class NavalPropulsion(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        df = pd.read_csv('/home/nsarna/data/naval_propulsion/data_cleaned.csv')

        self.label = df["Turbine_decay"].to_numpy()
        df = df.drop(["Turbine_decay"],axis=1)
        self.features = df.to_numpy()

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
        return self.helper_routines.return_label_feature(idx,self.label,self.features)

# combine cycle power plant dataset
class CCPP(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        df = pd.read_csv('/home/nsarna/data/combined_cycle_power_plant/cleaned_data.csv')

        self.label = df["PE"].to_numpy()
        df = df.drop(["PE"],axis=1)
        self.features = df.to_numpy()

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
         return self.helper_routines.return_label_feature(idx,self.label,self.features)

# combine cycle power plant dataset
class WineRed(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        df = pd.read_csv('/home/nsarna/data/winequality-red.csv',sep = ';')

        self.label = df["quality"].to_numpy()
        df = df.drop(["quality"],axis=1)
        self.features = df.to_numpy()

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
         return self.helper_routines.return_label_feature(idx,self.label,self.features)

class WineWhite(Dataset):

    def __init__(self,transform=transforms.ToTensor()):
        df = pd.read_csv('/home/nsarna/data/winequality-white.csv',sep = ';')

        self.label = df["quality"].to_numpy()
        df = df.drop(["quality"],axis=1)
        self.features = df.to_numpy()

        self.helper_routines = helper_routines()

        self.features = self.helper_routines.standardize_data(self.features)

    def __len__(self):
        return len(self.label)


    def __getitem__(self,idx):
         return self.helper_routines.return_label_feature(idx,self.label,self.features)




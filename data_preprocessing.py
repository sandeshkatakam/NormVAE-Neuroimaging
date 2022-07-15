import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# parser = argparse.ArgumentParser()
# # parser.add_argument("--cols", default=7, type=float) # Number of cols
# # parser.add_argument("--path", default="ADNI_sheet_for_VED.xlsx", type=string) # Path to dataset
# args = parser.parse_args()


path = "ADNI_sheet_for_VED.xlsx"
def load_and_standardize_data(path):
    df = pd.read_excel(path)
    healthy_indexes = df.index[df['CDGLOBAL'] == 0].tolist()
    healthy_df = df[df.index.isin(healthy_indexes)]
    healthy_normalized = healthy_df[["Normalised_Left_HIPPO","Normalised_Right_HIPPO", "Normalised_GM", "Normalised_WM", "Normalised_WMH", "Normalised_CSF", "Normalised_HIPPO"]].copy()
    healthy_normalized_df = healthy_normalized.reset_index()
    healthy_normalized_df = healthy_normalized_df.drop(["index"], axis =1)
    df = healthy_normalized_df
    # read in from csv
    #df = pd.read_csv(path, sep=',')
    # replace nan with -99
    df = df.fillna(-99)
    df = df.values.reshape(-1, df.shape[1]).astype('float32')
    # randomly split
    X_train, X_test = train_test_split(df, test_size=0.3, random_state=42)
    # standardize values
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)   
    return X_train, X_test, scaler


from torch.utils.data import Dataset, DataLoader

DATA_PATH = "ADNI_sheet_for_VED.xlsx"
class DataBuilder(Dataset):
    def __init__(self, path, train=True):
        self.X_train, self.X_test, self.standardizer = load_and_standardize_data(DATA_PATH)
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.len=self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.len=self.x.shape[0]
        del self.X_train
        del self.X_test 
    def __getitem__(self,index):      
        return self.x[index]
    def __len__(self):
        return self.len
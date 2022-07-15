from data_preprocessing import load_and_standardize_data, DataBuilder, DataLoader
from model import customLoss, Autoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument("--bsize", default=1024, type=int) # Batch size
parser.add_argument("--epochs", default=1500, type=int) # Number of epochs
parser.add_argument("--gensamples", default=10, type=int) # Number of samples
parser.add_argument("--output_format", default = "csv", type = str) # Ouput_format of the reconstructed samples

args = parser.parse_args()





X_train, X_test, scaler = load_and_standardize_data("ADNI_sheet_for_VED.xlsx")
DATA_PATH = "ADNI_sheet_for_VED.xlsx"
traindata_set=DataBuilder(DATA_PATH, train=True)
testdata_set=DataBuilder(DATA_PATH, train=False)

trainloader=DataLoader(dataset=traindata_set,batch_size=args.bsize)
testloader=DataLoader(dataset=testdata_set,batch_size=args.bsize)

epochs = args.epochs # Number of epochs to train
D_in = 7
H = 50
H2 = 12
log_interval = 50 
val_losses = []
train_losses = []
test_losses = []
loss_mse = customLoss()
model = Autoencoder(D_in, H, H2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch % 200 == 0:        
        print('====> Epoch: {} Average training loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))


def test(epoch):
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(testloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_mse(recon_batch, data, mu, logvar)
            test_loss += loss.item()
            if epoch % 200 == 0:        
                print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(testloader.dataset)))
            test_losses.append(test_loss / len(testloader.dataset))


for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)
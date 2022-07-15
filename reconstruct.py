import torch
import pandas as pd
from torch import nn, optim
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from train_model import trainloader, testloader
from model import Autoencoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gensamples", default=10, type=int) # Number of samples
parser.add_argument("--bsize", default=1024, type=int) # Batch size
parser.add_argument("--epochs", default=1500, type=int) # Number of epochs
parser.add_argument("--output_format", default = "xlsx", type = str) # Ouput_format of the reconstructed samples
args = parser.parse_args()

D_in = trainloader.dataset.x.shape[1]
H = 50
H2 = 12
n = args.gensamples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder(D_in, H, H2).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

with torch.no_grad():
    for batch_idx, data in enumerate(testloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)


scaler = trainloader.dataset.standardizer
recon_row = scaler.inverse_transform(recon_batch[0:n].cpu().numpy())
#real_row = scaler.inverse_transform(testloader.dataset.x[0:n].cpu().numpy())


reconstruct_df = pd.DataFrame(recon_row)
reconstruct_df.columns = ["Normalised_Left_HIPPO","Normalised_Right_HIPPO", "Normalised_GM","Normalised_WM","Normalised_WMH","Normalised_CSF", "Normalised_HIPPO"]
if args.output_format == "csv":
    reconstruct_df.to_csv("reconstructed_data.csv", index = False)
elif args.output_format == "xlsx":
    reconstruct_df.to_excel("reconstructed_data.xlsx", index = False)
        
from typing import Dict, Any
import hashlib
import joblib
import time
import shutil
from tqdm import tqdm
import copy
import gc,os,h5py,json,datetime,numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

import optuna
from optuna.pruners import BasePruner
from optuna.trial import TrialState
from optuna.storages import RetryFailedTrialCallback


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn import preprocessing

# for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False  # Disable cuDNN use of nondeterministic algorithms

from data import SpectraDataset


class cVAE(nn.Module):
    def __init__(self,dim_code):
        super().__init__()

        # self.dim_code = dim_code
        # encoder

        # self.label = nn.Embedding(10, dim_code)
        #
        #
        # self.en_Conv2d_1 = nn.Conv2d(
        #     1, 64, kernel_size=4, stride=2, padding=1
        # )
        # self.en_Conv2d_2 = nn.Conv2d(
        #     64, 128,kernel_size=4, stride=2, padding=1
        # )
        # self.en_ReLU = nn.ReLU()
        # self.en_BatchNorm = nn.BatchNorm2d(128)

        # [128,1,28,28] -> [128, 128, 7, 7]
        self.encoder = nn.Sequential(
            # output ( 14 as (28-4+(2*1))/2 + 1 )
            nn.Conv2d(1, 64,
                      kernel_size=4, stride=2, padding=1), # 28 - kernel_size=4 + (2 * padding=1) / stride=2 + 1
            nn.ReLU(),
            # output (7 is (14-4+(2*1))/2 + 1 )
            nn.Conv2d(64, 128,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # get the putput shape
        # encoder_shape =self.encoder(torch.empty(1,
        #                               image_shape[0], # colors / channels
        #                               image_shape[1], # X (width)
        #                               image_shape[2]  # Y (hight)
        #                               )
        #                   ).shape

        self.encoder_outshape = ()

        self.flatten_mu = nn.Linear(in_features=128* 32* 16, # 6272 From encoder output
                                    out_features=dim_code)
        self.flatten_logsigma = nn.Linear(in_features=128* 32* 16,
                                          out_features=dim_code)

        # decoder

        self.decode_linear = nn.Linear(in_features=2*dim_code,
                                       out_features=128 * 32* 16) # 6272
        self.decode_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                           kernel_size=4, stride=2, padding=1)
        self.decode_1 = nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                           kernel_size=4, stride=2, padding=1)


    def encode(self, x, y):
        # y = self.label(y) # [128,] -> [128, 4]

        # x = self.en_Conv2d_1(x) # [128,1,28,28] -> [128, 64, 14, 14] # 64-channels (14 is (28-4+(2*1))/2 + 1
        # x = self.en_ReLU(x) #
        # x = self.en_Conv2d_2(x) # [128, 64, 14, 14] -> [128, 128, 7, 7] # 128-channels (7 is (14-4+(2*1))/2 + 1
        # x = self.en_BatchNorm(x)
        # x = self.en_ReLU(x)

        x = self.encoder(x) # [64, 1, 128, 64] -> [64, 128, 32, 16]
        x = x.view(x.size(0), -1) # [64, 128, 32, 16] -> [64, 65536] # flattened
        mu, logsigma = self.flatten_mu(x), self.flatten_logsigma(x) # [128, 6272] -> [128, 4]
        x = self.gaussian_sampler(mu, logsigma) # [128, 4]
        z = torch.cat((x, y), dim=1) # [64, 9] -> [128, 18]
        return (z, x, mu, logsigma)

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = torch.exp(logsigma / 2)
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x, y):
        # y = self.label(y)
        z = torch.cat((x, y), dim=1)
        x = self.decode_linear(z)
        x = x.view(x.size(0), 128, 32, 16)
        x = F.relu(self.decode_2(x))
        reconstruction = F.sigmoid(self.decode_1(x))
        return reconstruction

    def forward(self, x, y):
        """ x : [128,1,28,28] y : [128,]"""

        # ENCODER
        _, x, mu, logsigma = self.encode(x, y)
        # y = self.label(y) # [128,] -> [128, 4]
        # x = self.encoder(x) # [128,] -> [128, 6272]
        # x = x.view(x.size(0), -1)
        # mu, logsigma = self.flatten_mu(x), self.flatten_logsigma(x)
        # x = self.gaussian_sampler(mu, logsigma)
        # z = torch.cat((x, y), dim=1)

        # DECODER
        reconstruction = self.decode(x, y)
        # x = self.decode_linear(z)
        # x = x.view(x.size(0), 128, 7, 7)
        # x = F.relu(self.decode_2(x))
        # reconstruction = F.sigmoid(self.decode_1(x))
        return (mu, logsigma, reconstruction)

def KL_divergence(mu, logsigma):
    loss = -0.5 * torch.sum(1.0 + logsigma - mu.pow(2) - logsigma.exp())
    return loss
def log_likelihood(x, reconstruction):
    loss = nn.BCELoss(reduction='sum')
    return loss(reconstruction, x)
def loss_vae(x, mu, logsigma, reconstruction):
    return KL_divergence(mu, logsigma) + log_likelihood(x, reconstruction)
class SpectraCVAE:
    def __init__(self,data_dir:str):
        self.device : torch.device
        self.data = SpectraDataset(working_dir=data_dir)
        self.data.load_and_normalize_data(
            data_dir=data_dir,limit=None,fname_x="X.h5",fname_y="Y.h5",add_color_channel=True
        )
        print(self.data.X.shape, self.data.y.shape)

    def plot_output_cvae(self, model, test_loader, epoch, epochs,
                         train_loss, val_loss, size=5):

        # clear_output(wait=True)
        plt.figure(figsize=(18, 6))
        for k in range(size):
            ax = plt.subplot(2, size, k + 1)
            img, label, _, _ = next(iter(test_loader))
            img = img.to(self.device)
            label = label.to(self.device)
            model.eval()
            with torch.no_grad():
                mu, logsigma, reconstruction = model(img, label)

            plt.imshow(img[k].cpu().squeeze().numpy().T, cmap='gray')
            plt.axis('off')
            plt.title(f'Real val {label[k].cpu().squeeze().numpy()}')
            ax = plt.subplot(2, size, k + 1 + size)
            plt.imshow(reconstruction[k].cpu().squeeze().numpy().T, cmap='gray')
            plt.axis('off')

            if k == size // 2:
                ax.set_title('Output')
        plt.suptitle('%d / %d - loss: %f val_loss: %f' % (epoch + 1, epochs, train_loss, val_loss))
        plt.show()

    def train_epoch_cvae(self, model, criterion, optimizer, data_loader):
        train_losses_per_epoch = []
        model.train()
        for x_batch, y, _, _ in data_loader:
            mu, logsigma, reconstruction = model(x_batch.to(self.device), y.to(self.device))
            loss = criterion(x_batch.to(self.device).float(), mu, logsigma, reconstruction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses_per_epoch.append(loss.item())

        return np.mean(train_losses_per_epoch), mu, logsigma, reconstruction

    def eval_epoch_cvae(self, model, criterion, optimizer, data_loader):
        val_losses_per_epoch = []
        model.eval()
        with torch.no_grad():
            for x_val, y, _, _ in data_loader:
                mu, logsigma, reconstruction = model(x_val.to(self.device), y.to(self.device))
                loss = criterion(x_val.to(self.device).float(), mu, logsigma, reconstruction)
                val_losses_per_epoch.append(loss.item())
        return np.mean(val_losses_per_epoch), mu, logsigma, reconstruction

    def __call__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            print("Running on GPU")
        else:
            print("Running on CPU")

        criterion = loss_vae
        autoencoder_cvae = cVAE(len(self.data.feature_names)).to(self.device)
        optimizer = torch.optim.Adam(autoencoder_cvae.parameters(), lr=5e-4)
        epochs = 30

        loss = {'train_loss': [], 'val_loss': []}

        train_loader, test_loader = self.data.get_dataloader(test_split=0.2, batch_size=64)
        with tqdm(desc="epoch", total=epochs) as pbar_outer:
            for epoch in range(epochs):
                print('* Epoch %d/%d' % (epoch + 1, epochs))
                train_loss, mu, logsigma, reconstruction = self.train_epoch_cvae(
                    autoencoder_cvae, criterion, optimizer, train_loader
                )
                val_loss, mu, logsigma, reconstruction = self.eval_epoch_cvae(
                    autoencoder_cvae, criterion, optimizer, test_loader
                )
                pbar_outer.update(1)
                loss['train_loss'].append(train_loss)
                loss['val_loss'].append(val_loss)
                self.plot_output_cvae(
                    autoencoder_cvae, test_loader, epoch, epochs, train_loss, val_loss, size=5)

        plt.figure(figsize=(15, 6))
        plt.semilogy(loss['train_loss'], label='Train')
        plt.semilogy(loss['val_loss'], label='Valid')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.legend()
        plt.title('Loss_vae')
        plt.show()


if __name__ == '__main__':
    working_dir = os.getcwd()+'/'
    cvae = SpectraCVAE(data_dir=working_dir)
    cvae()
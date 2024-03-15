"""
# Create Neural Network

### Primary Sources:
- [Paper by Lukosiute](https://arxiv.org/pdf/2204.00285.pdf) with [GitHub code](https://github.com/klukosiute/kilonovanet) using [Bulla's data](https://github.com/mbulla/kilonova_models/tree/master/bns/bns_grids/bns_m3_3comp).
- [PELS-VAE](https://github.com/jorgemarpa/PELS-VAE) github that had was used to draft train part for Lukosiute net ([data for it](https://zenodo.org/records/3820679#.XsW12RMzaRc))

### Secondary Sources
- [Tronto Autoencoder](https://www.cs.toronto.edu/~lczhang/aps360_20191/lec/w05/autoencoder.html) (Convolutional net)
- [Video with derivations](https://www.youtube.com/watch?v=iwEzwTTalbg)
- [Data sampling with scikit](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)
- [Astro ML BOOK repo with code](https://github.com/astroML/astroML_figures/blob/742df9181f73e5c903ea0fd0894ad6af83099c96/book_figures/chapter9/fig_sdss_vae.py#L45)

rsync -arvP --append ./{optuna_train.py,model_cvae.py,X.h5,Y.h5} vnedora@urash.gw.physik.uni-potsdam.de:/home/enlil/vnedora/work/cvae_optuna/

"""

from typing import Dict, Any
import hashlib
import joblib
import time

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

class SpectraDataset(Dataset):
    """
    LightCurve dataset
    Dispatches a lightcurve to the appropriate index
    """
    def __init__(self):
        self.fname_x = "X_afgpy.h5"
        self.fname_y = "Y_afgpy.h5"

    def load_normalize_data(self, data_dir, spectra_transform_method="minmax", limit=None):
        self._load_preprocessed_dataset(data_dir, limit)
        self._normalize_data(spectra_transform_method)

    def _load_preprocessed_dataset(self, data_dir, limit=None):
        # Load Prepared data
        with h5py.File(data_dir+self.fname_x,"r") as f:
            self.specs = np.array(f["X"]) #
            self.times = np.array(f["times"])
            self.freqs = np.array(f["freqs"])
        with h5py.File(data_dir+self.fname_y,"r") as f:
            self.pars = np.array(f["Y"])
            self.features_names = list([str(val.decode("utf-8")) for val in f["keys"]])

        if not limit is None:
            print(f"LIMITING data to {limit}")
            self.specs = self.specs[:limit] # [i_par_set, i_time, i_freq]
            self.pars = self.pars[:limit] # [i_par_set, i_par]

        print(f"specs={self.specs.shape}, pars={self.pars.shape}, times={self.times.shape} freqs={self.freqs.shape}")
        print(f"specs: min={self.specs.min()}, max={self.specs.max()}, pars: min={self.pars.min()} "
              f" max={self.pars.max()}, times={self.times.shape} freqs={self.freqs.shape}")
        print(self.features_names)
        assert self.pars.shape[0] == self.specs.shape[0], "size mismatch between lcs and pars"
        self.len = len(self.specs)

    def _normalize_data(self, spec_transform_method="minmax"):

        # scale parameters
        self.scaler = preprocessing.MinMaxScaler()
        self.scaler.fit(self.pars)
        self.pars_normed = self.scaler.transform(self.pars)

        # inverse transform
        # inverse = scaler.inverse_transform(normalized)
        if np.min(self.pars_normed) < 0. or np.max(self.pars_normed) > 1.01:
            raise ValueError(f"Parameter normalization error: "
                             f"min={np.min(self.pars_normed)} max={np.max(self.pars_normed)}")

        # preprocess spectra
        self.spec_transform_method = spec_transform_method
        self.spec_log_norm = self._transform_spectra(self.specs)
        if np.min(self.spec_log_norm) < 0. or np.max(self.spec_log_norm) > 1.01:
            raise ValueError(f"Spectra normalization error: min={np.min(self.spec_log_norm)} max={np.max(self.spec_log_norm)}")

    def __getitem__(self, index):
        """ returns image/spectrum, vars(params)[normalized], vars(params)[physical] """
        return (torch.from_numpy(self.spec_log_norm[index]).to(torch.float), # .to(self.device)
                torch.from_numpy(self.pars_normed[index]).to(torch.float),  # .to(self.device) .reshape(-1,1)
                self.specs[index],
                self.pars[index])


    def __len__(self):
        return len(self.specs)

    def _transform_spectra(self, specs):
        # convert flux densities to log10(flux densities)
        log_spectra = np.log10(specs)

        self.log_spec_min = log_spectra.min()
        self.log_spec_max = log_spectra.max()

        if (self.spec_transform_method=="minmax"):
            #self.lcs_log_norm = (log_lcs - np.min(log_lcs)) / (np.max(log_lcs) - np.min(log_lcs))
            self.spec_scaler = preprocessing.MinMaxScaler(feature_range=(0.0001, 0.9999)) # Otherwise max>1.000001

        elif (self.spec_transform_method=="standard"):
            self.spec_scaler = preprocessing.StandardScaler()

        self.spec_scaler.fit(log_spectra)
        return self.spec_scaler.transform(log_spectra)

    def inverse_transform_lc_log(self, spectra_log_normed):
        #return np.power(10, lcs_log_normed * (self.lc_max - self.lc_min) + self.lc_min)
        return np.power(10., self.spec_scaler.inverse_transform(spectra_log_normed))

    def _transform_pars(self, _pars):
        # print(_pars.shape, self.pars[:,0].shape)
        for i, par in enumerate(_pars.flatten()):
            if (par < self.pars[:,i].min()) or (par > self.pars[:,i].max()):
                raise ValueError(f"Parameter '{i}'={par} is outside of the training set limits "
                                 f"[{self.pars[:,i].min()}, {self.pars[:,i].max()}]")
        return self.scaler.transform(_pars)

    def _invert_transform_pars(self, _pars):
        self.scaler.inverse_transform(_pars)

    def get_dataloader(self, test_split=0.2, batch_size=32):
        """
        If
        :param batch_size: if 1 it is stochastic gradient descent, else mini-batch gradient descent
        :param test_split:
        :return:
        """
        dataset_size = len(self)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.valid_sampler = SubsetRandomSampler(test_indices)
        train_loader = DataLoader(self, batch_size=batch_size,
                                  sampler=self.train_sampler, drop_last=False)
        test_loader = DataLoader(self, batch_size=batch_size,
                                 sampler=self.valid_sampler, drop_last=False)

        return (train_loader, test_loader)


if __name__ == '__main__':
    data = SpectraDataset()
    data.load_normalize_data(os.getcwd()+'/')
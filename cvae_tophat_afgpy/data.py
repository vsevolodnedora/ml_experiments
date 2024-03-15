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


def preprocess_dataset(data_dir=os.getcwd()+'/', fname_x = "X_afgpy.h5", fname_y = "Y_afgpy.h5"):
    with h5py.File(data_dir+fname_x,"r") as f:
        specs = np.array(f["X"])
        times = np.array(f["times"])
        freqs = np.array(f["freqs"])
    with h5py.File(data_dir+fname_y,"r") as f:
        pars = np.array(f["Y"])
        feature_names = list([str(val.decode("utf-8")) for val in f["keys"]])

    # check for nans in data
    if (np.any(np.isnan(specs))):
        raise ValueError("Nans in spectra")
    if (np.any(np.isnan(pars))):
        raise ValueError("Nans in pars")

    # prevent spectrum from being 0. (cannot use log scaling)
    if np.min(specs) <= 0.:
        val = np.min(specs[specs > 0.])
        specs[specs <= 0] = val
        print(f"Warning overwriting spectra to mask zeroes. New min = {val}")
        with h5py.File(data_dir+fname_x,"w") as f:
            f.create_dataset(name="X", data=specs)
            f.create_dataset(name="times", data=times)
            f.create_dataset(name="freqs", data=freqs)



def prepare_dataset_scalers(data_dir=os.getcwd()+'/', out_dir=os.getcwd()+'/',
                            fname_x = "X_afgpy.h5", fname_y = "Y_afgpy.h5"):

    """
    Creates files:
        train_data_meta.h5
        y_scaler.pkl
        x_scaler.json
        train_data_info.json
    :param data_dir:
    :param out_dir:
    :param fname_x:
    :param fname_y:
    :return:
    """

    with h5py.File(data_dir+fname_x,"r") as f:
        specs = np.array(f["X"])
        times = np.array(f["times"])
        freqs = np.array(f["freqs"])
    with h5py.File(data_dir+fname_y,"r") as f:
        pars = np.array(f["Y"])
        feature_names = list([str(val.decode("utf-8")) for val in f["keys"]])
    print(f"times={times.shape} freq={freqs.shape}")

    assert pars.shape[0] == specs.shape[0], "size mismatch between lcs and pars"

    # load train data meta
    fname_meta = data_dir+"train_data_meta.h5"
    print(f"Saving... {fname_meta}")
    with h5py.File(fname_meta, "w") as f:
        f.create_dataset(name="times", data=times)
        f.create_dataset(name="freqs", data=freqs)

    # scale parameters (features)
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0.001,0.999))
    scaler_y.fit(pars)
    fname_y = out_dir+'y_scaler.pkl'
    print(f"Saving...{fname_y}")
    joblib.dump(scaler_y, fname_y)

    # scale features
    fname_x = out_dir+"x_scaler.json"
    print(f"Saving... {fname_x}")
    with open(fname_x, 'w') as outfile:
        json.dump(
            {
                "type":"log10_minmax",
                "log_min": str(np.min(np.log10(specs))),
                "log_max": str(np.max(np.log10(specs))),
            },
            outfile)



    with open(out_dir+"train_data_info.json", 'w') as outfile:
        json.dump(
            {
                "target": "spectrum",
                "y_scaler": fname_y,
                "x_scaler": fname_x,
                "feature_names":feature_names,
            },
            outfile)
    

class SpectraDataset(Dataset):

    def __init__(self, working_dir):
        """
        Requires Files:
            train_data_meta.h5
            y_scaler.pkl
            x_scaler.json
            train_data_info.json
        Optional (for training)
            X.h5
            Y.h5
        :param working_dir:
        """
        self.working_dir = working_dir

        # load training data information
        with open(self.working_dir+"train_data_info.json", 'r') as infile:
            self.info = json.load(infile)
        self.feature_names = self.info['feature_names']

        # create yscaler
        if (self.info["y_scaler"].__contains__(".pkl")):
            scaler_y = joblib.load(self.info["y_scaler"])
            self.transform_y = lambda y: scaler_y.transform(y)
            self.inverse_transform_y = lambda y_norm: scaler_y.inverse_transform(y_norm)

        elif (self.info["y_scaler"] == "none"):
            self.transform_y = lambda y: y
            self.inverse_transform_y = lambda y_norm: y_norm
        else: raise NotImplementedError("Not implemented yet")

        # read xscaler
        if (self.info["x_scaler"].__contains__(".json")):
            # load the file describing the data normalization
            with open(self.info["x_scaler"]) as json_data:
                x_scaler_dict = json.load(json_data)
                json_data.close()
            if x_scaler_dict["type"] == "log10_minmax":
                log_min, log_max = np.float64(x_scaler_dict["log_min"]), np.float64(x_scaler_dict["log_max"])
                self.transform_X = lambda X: (np.log10(X) - log_min) / (log_max - log_min)
                self.inverse_transform_X = lambda X_norm: np.power(10, X_norm * (log_max - log_min) + log_min)
            else: raise NotImplementedError("Not implemented yet")
        else: raise NotImplementedError("Not implemented yet")

        # load train data meta (grid for X)
        with h5py.File(self.working_dir+"train_data_meta.h5", "r") as f:
            self.times = np.array(f["times"])
            self.freqs = np.array(f["freqs"])


    def load_and_normalize_data(self, data_dir=None or str, limit=None,
                                fname_x = "X_afgpy.h5", fname_y = "Y_afgpy.h5",
                                add_color_channel=False):
        if data_dir is None: data_dir = self.working_dir
        # Load Prepared data
        with h5py.File(data_dir+fname_x,"r") as f:
            self.X = np.array(f["X"]) if limit is None else np.array(f["X"])[:limit] # spectra [i_example, i_time, i_freq]
            self.times = np.array(f["times"])
            self.freqs = np.array(f["freqs"])
            if add_color_channel:
                # add color channel for grayscale image
                self.X = np.expand_dims(self.X, axis=1)
        with h5py.File(data_dir+fname_y,"r") as f:
            self.y = np.array(f["Y"]) if limit is None else np.array(f["Y"])[:limit] # labels [i_example, i_label]
        if not limit is None:
            print(f"Warning. Limititing training data to {len(self.X)} examples")
        # normalize data
        self.y_norm = self.transform_y(self.y)
        self.X_norm = self.transform_X(self.X)

    def __len__(self):
        return len(self.X_norm)
    def __getitem__(self, index):
        """ returns image/spectrum, vars(params)[normalized], vars(params)[physical] """
        return (torch.from_numpy(self.X_norm[index]).to(torch.float), # .to(self.device)
                torch.from_numpy(self.y_norm[index]).to(torch.float),  # .to(self.device) .reshape(-1,1)
                self.X[index],
                self.y[index])
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
    preprocess_dataset(data_dir=os.getcwd()+'/',
                       fname_x = "X_afgpy.h5",
                       fname_y = "Y_afgpy.h5")

    prepare_dataset_scalers(data_dir=os.getcwd()+'/',
                            out_dir=os.getcwd()+'/',
                            fname_x = "X_afgpy.h5",
                            fname_y = "Y_afgpy.h5")

    data = SpectraDataset(working_dir=os.getcwd()+'/')
    data.load_and_normalize_data(data_dir=None, limit=None, fname_x = "X_afgpy.h5", fname_y = "Y_afgpy.h5")

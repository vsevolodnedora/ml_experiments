from typing import Dict, Any
import hashlib
import joblib
import time
import shutil

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

from model import CVAE
from data import SpectraDataset

class EarlyStopping:
    """Early stops the training if validation loss doesn't
        improve after a given patience."""
    def __init__(self, pars, verbose):
        """
        Attributes
        ----------
        patience  : int
            How long to wait after last time validation loss improved.
            Default: 7
        min_delta : float
            Minimum change in monitored value to qualify as
            improvement. This number should be positive.
            Default: 0
        verbose   : bool
            If True, prints a message for each validation loss improvement.
            Default: False
        """
        self.verbose = verbose
        self.patience = pars["patience"]
        self.verbose = pars["verbose"]
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.min_delta = pars["min_delta"]

    def __call__(self, val_loss):

        current_loss = val_loss

        if self.best_score is None:
            self.best_score = current_loss
        elif abs(current_loss - self.best_score) < self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = current_loss
            self.counter = 0
        return False

def select_optimizer(model, pars:dict)->torch.optim:
    if (pars["name"]=="Adam"):
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=pars["lr"],
                                     eps=pars["eps"],
                                     weight_decay=pars["weight_decay"])
    elif (pars["name"]=="SGD"):
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=pars["lr"],
                                    momentum=pars["momentum"],
                                    weight_decay=pars["weight_decay"])
    elif (pars["name"]=="RMSprop"):
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=pars["lr"],
                                        momentum=pars["momentum"],
                                        eps=pars["eps"],
                                        weight_decay=pars["weight_decay"],
                                        alpha=pars["alpha"])
    else:
        raise NameError("Optimizer is not recognized")
    return optimizer

def select_scheduler(optimizer, pars:dict)->optim.lr_scheduler or None:
    """lr_sch='step'"""
    lr_sch = pars["name"]
    del pars["name"]
    if lr_sch == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=pars["step_size"], gamma=pars["gamma"])
    elif lr_sch == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=pars["gamma"])
    elif lr_sch == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=pars["T_max"])
    elif lr_sch == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=pars["mode"],factor=pars["factor"])
    else:
        scheduler = None
    return scheduler

def select_model(device, pars, verbose):

    name = pars["name"]
    del pars["name"]
    if name == CVAE.name:
        model = CVAE(**pars)
    else:
        raise NameError(f"Model {name} is not recognized")

    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f'Num of trainable params: {n_train_params}')

    if torch.cuda.device_count() > 1 and True:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    # model = torch.compile(model, backend="aot_eager").to(device)

    return model

def beta_scheduler(beta, epoch, beta0=0., step=50, gamma=0.1):
    """Scheduler for beta value, the sheduler is a step function that
    increases beta value after "step" number of epochs by a factor "gamma"

    Parameters
    ----------
    epoch : int
        epoch value
    beta0 : float
        starting beta value
    step  : int
        epoch step for update
    gamma : float
        linear factor of step scheduler

    Returns
    -------
    beta
        beta value
    """
    if beta == 'step':
        return np.float32( beta0 + gamma * (epoch+1 // step) )
    else:
        return np.float32(beta)

# class Loss:
#     def __init__(self, pars, device, verbose):
#         self.device = device
#         self.verbose = verbose
#         self.pars = pars
#     def __call__(self, x, recon_x, mu, logvar, beta):
#         pars = self.pars
#
#
#
#         if pars["mse_or_bce"]=="mse":   base = F.mse_loss(recon_x, x.view(-1, recon_x.shape[1]), reduction=pars["reduction"])
#         elif pars["mse_or_bce"]=="bce": base = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction=pars["reduction"])
#         else: base = 0
#
#         kld_l = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp())
#
#         if pars["kld_norm"]: kld_l = kld_l / x.shape[0]
#         if pars["use_beta"]: kld_l *= beta
#
#         # loss = base + kld_l
#
#
#         # if self.verbose and not torch.isfinite(base):
#         #     print(f"Error in base loss value: {loss.item()} base={base.item()} kld_norm={kld_l.item()} beta={beta}")
#         #     base = torch.tensor(1.e3, dtype=torch.float32).to(self.device)
#         # if self.verbose and not torch.isfinite(kld_l):
#         #     print(f"Error in KL loss value: {loss.item()} base={base.item()} kld_norm={kld_l.item()} beta={beta}")
#         #     kld_l = torch.tensor(1.e3, dtype=torch.float32).to(self.device)
#
#         return base + kld_l

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.shape[1]), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()




def OLD_inference(y_vec:list, model:CVAE, dataset:SpectraDataset, device):
    # if len(pars) != model.z_dim:
    #     raise ValueError(f"Number of parameters = {len(pars)} does not match the model latent space size {model.z_dim}")
    # create state vector for intput data (repeat physical parameters for times needed)
    y_vec = np.asarray(y_vec).reshape(1, -1)
    # normalize parameters as in the training data
    y_vec_norm = dataset.transform_y(y_vec)
    # generate prediction
    with torch.no_grad():
        # convert intput data to the format of the hidden space
        z = (torch.zeros((1, model.z_dim)).repeat((len(y_vec_norm), 1)).to(device).to(torch.float))
        # create the input for the decoder
        decoder_input = torch.cat((z, torch.from_numpy(y_vec_norm).to(device).to(torch.float)), dim=1)
        # perform reconstruction using model
        reconstructions = model.decoder(decoder_input)
    # move prediction to cpu and numpy
    reconstructions_np = reconstructions.double().cpu().detach().numpy()
    # undo normalization that was done in training data
    spectra_nn = dataset.inverse_transform_X(reconstructions_np)
    return spectra_nn

def plot_lcs(tasks, model, device, dataset, model_dir):

    freqs = np.unique(dataset.pars[:,dataset.features_names.index("freq")])
    norm = LogNorm(vmin=np.min(freqs),
                   vmax=np.max(freqs))
    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    fig, axes = plt.subplots(2,3, figsize=(15,5), sharex='all',
                             gridspec_kw={'height_ratios': [3, 1]})
    for i, task in enumerate(tasks):
        req_pars = copy.deepcopy(task["pars"])
        # pars = [req_pars[feat] for feat in features_names]
        l2s = []
        for freq in freqs:

            # get light curve from training data
            mask = np.ones_like(dataset.pars[:,0],dtype=bool)
            req_pars["freq"] = freq
            for j, par in enumerate(dataset.features_names):
                i_mask = dataset.pars[:,j] == req_pars[par]
                if np.sum(i_mask) == 0:
                    raise ValueError(f"par={par} requested {req_pars[par]} "
                                     f"not found in data\n{np.unique(dataset.pars[:,j])}")
                mask = (mask & i_mask)
            if np.sum(mask) > 1:
                raise ValueError("error in extracting LC from train data")
            # expected array with one index
            lc = np.log10(np.array(dataset.lcs[mask, :])).flatten()



            l11, = axes[0, i].plot(dataset.times/86400, lc,
                                   ls='-', color='gray', alpha=0.5, label=f"Original") # for second legend
            l21, = axes[0, i].plot(dataset.times/86400, lc,
                                   ls='-', color=cmap(norm(freq)), alpha=0.5, label=f"{freq/1e9:.1f} GHz")
            l2s.append(l21)

            # get light curve from model
            # get list of parameters from dictionary
            req_pars_list = [np.float32(req_pars[feat]) for feat in dataset.features_names]
            lc_nn = np.log10(np.array(inference(req_pars_list, model, dataset, device)).flatten())
            l12, = axes[0, i].plot(dataset.times/86400, lc_nn,ls='--', color='gray', alpha=0.5, label=f"cVAE") # for second legend
            l22, = axes[0, i].plot(dataset.times/86400, lc_nn,ls='--', color=cmap(norm(freq)), alpha=0.5)


            # plot difference
            axes[1, i].plot(dataset.times/86400, lc-lc_nn, ls='-', color=cmap(norm(freq)), alpha=0.5)

        axes[0, i].set_xscale("log")
        axes[1, i].set_xlabel(r"times [day]")

    axes[0,0].set_ylabel(r'$\log_{10}(F_{\nu})$')
    axes[1,0].set_ylabel(r'$\Delta\log_{10}(F_{\nu})$')

    first_legend = axes[0,0].legend(handles=l2s, loc='upper right')

    axes[0,0].add_artist(first_legend)
    axes[0,0].legend(handles=[l11,l12], loc='lower right')

    plt.tight_layout()
    plt.savefig(model_dir+"/lcs.png",dpi=256)
    # plt.show()
    plt.close(fig)

def plot_violin(delta, dataset, model_dir):


    def find_nearest_index(array, value):
        ''' Finds index of the value in the array that is the closest to the provided one '''
        idx = (np.abs(array - value)).argmin()
        return idx

    cmap_name = "coolwarm_r" # 'coolwarm_r'
    cmap = plt.get_cmap(cmap_name)

    freqs = np.unique(dataset.pars[:,dataset.features_names.index("freq")])
    norm = LogNorm(vmin=np.min(freqs),
                   vmax=np.max(freqs))

    req_times = np.array([0.1, 1., 10., 100., 1000., 1e4]) * 86400.

    fig, axes = fig, ax = plt.subplots(2, 3, figsize=(12, 5), sharex="all", sharey="all")
    ax = ax.flatten()
    for ifreq, freq in enumerate(freqs):
        i_mask1 = dataset.pars[:, dataset.features_names.index("freq")] == freq

        _delta = delta[i_mask1]
        time_indeces = [find_nearest_index(dataset.times, t) for t in req_times]
        _delta = _delta[:, time_indeces]

        color = cmap(norm(freqs[0]))

        if np.sum(_delta) == 0:
            raise ValueError(f"np.sum(delta) == 0 delta={_delta.shape}")
        # print(_delta.shape)
        violin = ax[ifreq].violinplot(_delta, positions=range(len(req_times)),
                                      showextrema=False, showmedians=True)


        for pc in violin['bodies']:
            pc.set_facecolor(color)
        violin['cmedians'].set_color(color)
        for it, t in enumerate(req_times):
            ax[ifreq].vlines(it, np.quantile(_delta[:,it], 0.025), np.quantile(_delta[:,it], 0.975),
                             color=color, linestyle='-', alpha=.8)

        # ax[ifreq].hlines([-1,0,1], 0.1, 6.5, colors='gray', linestyles=['dashed', 'dotted', 'dashed'], alpha=0.5)


        ax[ifreq].set_xticks(np.arange(0, len(req_times)))
        # print(ax[ifreq].get_xticklabels(), ax[ifreq])
        _str = lambda t : '{:.1f}'.format(t/86400.) if t/86400. < 1 else '{:.0f}'.format(t/86400.)
        ax[ifreq].set_xticklabels([_str(t) for t in req_times])

        ax[ifreq].annotate(f"{freq/1e9:.1f} GHz", xy=(1, 1),xycoords='axes fraction',
                           fontsize=12, horizontalalignment='right', verticalalignment='bottom')


    # Create the new axis for marginal X and Y labels
    ax = fig.add_subplot(111, frameon=False)

    # Disable ticks. using ax.tick_params() works as well
    ax.set_xticks([])
    ax.set_yticks([])

    # Set X and Y label. Add labelpad so that the text does not overlap the ticks
    ax.set_xlabel(r"Time [days]", labelpad=20, fontsize=12)
    ax.set_ylabel(r"$\Delta \log_{10}(F_{\nu})$", labelpad=40, fontsize=12)
    plt.tight_layout()
    plt.savefig(model_dir+"violin.png",dpi=256)

    # plt.show()

def plot_loss(model, loss_df, model_dir):
    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(5,3))
    ax.plot(range(len(loss_df)), loss_df["train_losses"],ls='-',color='blue',label='training loss')
    ax.plot(range(len(loss_df)), loss_df["valid_losses"],ls='-',color='red',label='validation loss')
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(model_dir+"/loss.png",dpi=256)
    # plt.show()
    plt.close(fig)

def compute_error_metric(dataset, model, device, model_dir, verbose):

    # compute difference between all light curves and NN light curves
    nn_spectra = np.vstack((
        [inference(dataset.y[j, :], model, dataset, device) for j in range(len(dataset.y[:,0]))]
    ))
    # print(f"lcs={dataset.lcs.shape} nn_lcs={nn_lcs.shape}")
    log_lcs = np.log10(dataset.X)
    log_nn_lcs = np.log10(nn_spectra)
    delta = (log_lcs - log_nn_lcs)

    plot_violin(delta, dataset, model_dir)

    # return total error
    rmse = root_mean_squared_error(log_lcs, log_nn_lcs)
    if verbose:
        print("Total RMSE: {:.2e}".format(rmse))
    return rmse

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()

def make_dir_for_run(working_dir, trial, dict_, verbose):
    final_model_dir = f"{working_dir}/trial_{trial}/"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    else:
        shutil.rmtree(final_model_dir)
        os.makedirs(final_model_dir)
    if verbose:
        print("Saving pars in {}".format(final_model_dir))
    with open(final_model_dir+"pars.json", "w") as outfile:
        json.dump(dict_, outfile)
    return final_model_dir


class TrainCVAE:

    def __init__(self, dataset:SpectraDataset, is_optuna:bool):
        self.dataset = dataset
        self.is_optuna = is_optuna

    def __call__(self, trial:optuna.trial.Trial or None):
        # ------------------------------
        do = self.is_optuna
        verbose = True
        if do: print(f"TRIAL {trial.number}")
        # ------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print("Running on GPU")
        else:
            print("Running on CPU")
        # device="cpu"

        # =================== INIT ===================
        main_pars = {
            "batch_size":trial.suggest_int("batch_size", low=16, high=128, step=8) if do else 8,
            "epochs":150,
            "beta":"step"
        }

        #  feature_size=8192, hidden_dim=400, latent_size=20, class_size=7,
        model_pars = {"name":"CVAE",
                      "feature_size": len(dataset.times) * len(dataset.freqs),
                      "hidden_dim": trial.suggest_int("hidden_dim", low=20, high=1000, step=10) if do else 400,
                      "latent_size": trial.suggest_int("latent_size", low=4, high=64, step=4) if do else len(self.dataset.feature_names) ,
                      "class_size":len(self.dataset.feature_names),
                      "init_weights":True
                      }
        model = select_model(device, copy.deepcopy(model_pars), verbose)


        loss_pars = {
            "mse_or_bce":trial.suggest_categorical(name="mse_or_bce",choices=["mse","bce"]) if do else "mse",
            "reduction": trial.suggest_categorical(name="reduction",choices=["sum","mean"]) if do else "sum",
            "kld_norm":  trial.suggest_categorical(name="kld_norm",choices=[True,False]) if do else True,
            "use_beta":  trial.suggest_categorical(name="use_beta",choices=[True,False]) if do else True,
        }
        loss_cvae = loss_function


        optimizer_pars = {
            "name":    trial.suggest_categorical(name="optimizer", choices=["Adam", "SGD"]) if do else "Adam",
            "lr":      trial.suggest_float(name="lr", low=1.e-5, high=1.e-2, log=False) if do else 1.e-3,
            "momentum":trial.suggest_categorical(name="momentum", choices=[0.0001,0.001,0.01,0.1,0.]) if do else 0.,
            "eps":     trial.suggest_categorical(name="eps", choices=[0.0001,0.001,0.01,0.1,0.]) if do else 0.,
            "weight_decay":trial.suggest_categorical(name="weight_decay", choices=[0.0001,0.001,0.01,0.1,0.]) if do else 0.,
            "alpha":  trial.suggest_categorical(name="alpha", choices=[0.0001,0.001,0.01,0.1,0.]) if do else 0.,
        }
        optimizer = select_optimizer(model, copy.deepcopy(optimizer_pars))


        scheduler_pars = {
            "name":trial.suggest_categorical(name="scheduler",choices=["step","exp","cos","plateau"]) if do else "step",
            "step_size":trial.suggest_int(name="step_size", low=2, high=50, step=2) if do else 5,
            "gamma":trial.suggest_float(name="gamma", low=1.e-3, high=0.985, log=False) if do else 0.1,
            "T_max":trial.suggest_int(name="T_max", low=10, high=100,step=10) if do else 50,
            "mode":'min',"factor":.5
            # "name":"exp", "gamma":0.985,
            # "name":"cos", "T_max":50, "eta_min":1e-5,
            # "name":"plateau", "mode":'min', "factor":.5,"verbose":True
        }
        scheduler = select_scheduler(optimizer, copy.deepcopy(scheduler_pars))


        early_stopping_pars = {
            "patience":5,
            "verbose":True,
            "min_delta":0.
        }
        early_stopper = EarlyStopping(copy.deepcopy(early_stopping_pars),verbose)

        # ================== RUN ===================
        numer = trial.number if do else 0
        final_model_dir = make_dir_for_run(
            working_dir, numer,{
                **main_pars,**model_pars,**loss_pars,
                **optimizer_pars,**scheduler_pars,**early_stopping_pars
            }, verbose)

        main_pars["final_model_dir"] = final_model_dir
        main_pars["checkpoint_dir"] = None # final_model_dir

        # ==================== DATA ====================

        train_loader, valid_loader = self.dataset.get_dataloader(
            test_split=.2,
            batch_size=main_pars["batch_size"]
        )

        # =================== TRAIN & TEST ===================

        epochs = main_pars["epochs"]
        beta = main_pars["beta"] # or 'step'
        time_start = datetime.datetime.now()
        train_loss = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
        valid_loss  = {key: [] for key in ["KL_latent", "BCE", "MSE", "Loss"]}
        epoch, num_steps = 0, 0
        for epoch in range(epochs):
            e_time = datetime.datetime.now()
            if verbose:
                print(f"-----| Epoch {epoch}/{epochs} | Train/Valid {len(train_loader)}/{len(valid_loader)} |-------")

            beta = beta_scheduler(beta, epoch) # Scheduler for beta value

            # ------------- Train -------------
            model.train() # set model into training mode
            losses = []
            for i, (x_image, y_label, x_phys, y_phys) in enumerate(train_loader):
                num_steps += 1
                optimizer.zero_grad() # Resets the gradients of all optimized tensors
                recon_batch, mu, logvar = model(x_image.to(device), y_label.to(device)) # Forward pass only to get logits/output (evaluate model)
                if torch.any(torch.isnan(recon_batch)): # check if nans
                    raise optuna.exceptions.TrialPruned()
                loss = loss_cvae(recon_batch, x_image, mu, logvar) # compute/store loss
                loss.backward() # computes dloss/dx for every parameter x which has requires_grad=True.
                losses.append(loss.detach().cpu().numpy())
                optimizer.step() # perform a single optimization step
            train_loss['Loss'].append(np.sum(losses)/len(self.dataset.train_sampler))
            if verbose:
                print(f"\t Train loss: {train_loss['Loss'][-1]:.2e}")
            if (not np.isfinite(train_loss['Loss'][-1])):
                raise optuna.exceptions.TrialPruned()
            # ------------- Validate -------------
            model.eval()
            losses = []
            with torch.no_grad():
                for i, (x_image, y_label, x_phys, y_phys) in enumerate(valid_loader):
                    x_image = x_image.to(device)
                    y_label = y_label.to(device)
                    recon_batch, mu, logvar = model(x_image, y_label) # evaluate model on the data
                    if torch.any(torch.isnan(recon_batch)):
                        raise optuna.exceptions.TrialPruned()
                    loss = loss_cvae(recon_batch, x_image, mu, logvar) #  computes dloss/dx requires_grad=False
                    # mse_mean.append(np.sqrt(F.mse_loss(xhat, data, reduction="mean").item()))
                    # mse_sum.append(np.sqrt(F.mse_loss(xhat, data, reduction="sum").item()))
                    losses.append(loss.detach().cpu().numpy())

            # all_y = torch.from_numpy(dataset.lcs_log_norm).to(torch.float), # .to(self.device)
            # all_x = torch.from_numpy(dataset.pars_normed).to(torch.float)

            # xhat, _, _, _ = model(all_y, all_x) # evaluate model on the data
            # mse_sum_ = np.sqrt(F.mse_loss(xhat, all_x, reduction="sum").item())
            # mse_sum__ = np.sum(mse_sum)

            # valid_loss['Loss'].append(losses / len(valid_loader) * dataset.batch_size)
            # valid_mse["mean"].append(np.sum(mse_mean)/len(valid_loader))
            # valid_mse["sum"].append(np.sum(mse_sum)/len(valid_loader))
            valid_loss['Loss'].append(np.sum(losses) / len(self.dataset.valid_sampler))
            if (not np.isfinite(train_loss['Loss'][-1])):
                raise optuna.exceptions.TrialPruned()
            # ----------- Evaluate ------------
            # num_samples = 10
            # all_y = torch.from_numpy(dataset.lcs_log_norm).to(torch.float).to(device)
            # all_x = torch.from_numpy(dataset.pars_normed).to(torch.float).to(device)
            # model.eval()
            # y_pred = torch.empty_like(all_y)
            # with torch.no_grad():
            #     for k in range(len(all_x)):
            #         multi_recon = torch.empty((num_samples, len(dataset.lcs[0])))
            #         for i in range(num_samples):
            #             z = torch.randn(1, model.z_dim).to(device).to(torch.float)
            #             x = all_x[k:k+1].to(torch.float)
            #             z1 = torch.cat((z, x), dim=1)
            #             recon = model.decoder(z1)
            #             multi_recon[i] = recon
            #         mean = torch.mean(multi_recon, axis=0)
            #         y_pred[k] = mean
            # errors = (torch.sum(
            #     torch.abs(dataset.inverse_transform_lc_log(all_y)
            #               - dataset.inverse_transform_lc_log(y_pred)), axis=1)
            #           /torch.sum( dataset.inverse_transform_lc_log(all_y),axis=1))
            # torch.save(y_pred, './'+ARGS.exp_name+'/test_predictions.pt')
            # torch.save(errors, './'+ARGS.exp_name+'/test_epsilons.pt')

            if verbose:
                print(f"\t Valid loss: {valid_loss['Loss'][-1]:.2e}")
                # f"<RMSE> {valid_mse['mean'][-1]:.2e} sum(RMSE)={valid_mse['sum'][-1]:.2e}")

            # ------------- Update -------------
            if not (scheduler is None):
                if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss['Loss'][-1])
                else:
                    scheduler.step()

            epoch_time = datetime.datetime.now() - e_time
            elap_time = datetime.datetime.now() - time_start
            if verbose:
                print(f"\t Time={elap_time.seconds/60:.2f} m  Time/Epoch={epoch_time.seconds:.2f} s ")

            # ------------- Save chpt -------------
            if not main_pars["checkpoint_dir"] is None:
                fname = '%s_%d.chkpt' % (main_pars["checkpoint_dir"], epoch)
                if verbose: print("\t Saving checkpoint")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'beta': beta_scheduler(beta, epoch),
                    'train_losses':train_loss['Loss'][-1],
                    'valid_losses':valid_loss['Loss'][-1],
                    'train_batch': len(train_loader),
                    'test_batch':len(valid_loader)
                }, fname)

            # ------------- Stop if -------------
            if (early_stopper(valid_loss['Loss'][-1])):
                break

            # ------------- Prune -------------
            if do:
                trial.report(valid_loss['Loss'][-1], epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        # =================== SAVE ===================

        if not main_pars["final_model_dir"] is None:
            fname = main_pars["final_model_dir"] + "model.pt"
            if verbose: print(f"Saving model {fname}")
            torch.save({
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_loss,
                'valid_losses': valid_loss,
                'model_init':model_pars,
                'metadata':{
                    "batch_size":main_pars["batch_size"],
                    "beta":beta,
                    "epochs":epochs,
                    "last_epoch":epoch
                },
                "dataset":{
                    "x_transform":"minmax",
                    "y_transform":"log_minmax"
                }
            }, fname)

            if verbose: print(f"Saving loss history")
            res = {"train_losses":train_loss["Loss"], "valid_losses":valid_loss["Loss"]}
            pd.DataFrame.from_dict(res).to_csv(fname.replace('.pt','_loss_history.csv'), index=False)

        # ================== CLEAR ==========s=========

        model.apply(reset_weights)

        # ================= ANALYZE ================

        rmse = compute_error_metric(dataset, model, device, final_model_dir, verbose)
        if (not np.isfinite(rmse)):
            raise optuna.exceptions.TrialPruned()
        return rmse
        # return np.random.randn()


if __name__ == '__main__':

    working_dir = os.getcwd() + '/'

    dataset = SpectraDataset(working_dir)
    dataset.load_and_normalize_data(data_dir=None, limit=None,
                                    # fname_x = "X_afgpy.h5", fname_y = "Y_afgpy.h5"
                                    fname_x = "X.h5", fname_y = "Y.h5"
                                    )
    for i in range(dataset.y_norm.shape[1]):
        print(f"{dataset.feature_names[i]} : "
              f"[{np.min(dataset.y_norm[:,i])}, {np.max(dataset.y_norm[:,i])}]"
              f" N_unique = {len(np.unique(dataset.y_norm[:,i]))}")

    train = TrainCVAE(dataset, is_optuna=False)
    train(None)
    exit(0)

    # Create an Optuna study to maximize test accuracy
    # pruner = TimoutPruner(max_sec_per_trial=20.*60.)
    study = optuna.create_study(direction="minimize")
    # pruner=pruner)
    study.optimize(Objective(dataset),
                   n_trials=1,
                   callbacks=[lambda study, trial: gc.collect()])

    print("Study completed successfully. Saving study")
    fpath = working_dir + "study.pkl"
    joblib.dump(study, fpath)

    # Find number of pruned and completed trials
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    with open(working_dir+"summary.txt", 'w') as file:
        # Display the study statistics
        file.write("\nStudy statistics: \n")
        file.write(f"  Number of finished trials: {len(study.trials)}\n")
        file.write(f"  Number of pruned trials: {len(pruned_trials)}\n")
        file.write(f"  Number of complete trials: {len(complete_trials)}\n")

        trial = study.best_trial
        file.write("\nBest trial:\n")
        file.write(f"  Value: {trial.value}\n")
        file.write(f"  Numer: {trial.number}\n")
        file.write("  Params: \n")
        for key, value in trial.params.items():
            file.write("    {}: {}\n".format(key, value))

        # Find the most important hyperparameters
        most_important_parameters = optuna.importance.get_param_importances(study, target=None)
        # Display the most important hyperparameters
        file.write('\nMost important hyperparameters:\n')
        for key, value in most_important_parameters.items():
            file.write('  {}:{}{:.2f}%\n'.format(key, (15-len(key))*' ', value*100))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start',
                                        'datetime_complete',
                                        'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by loss):\n {}".format(df))


import copy

import afterglowpy as grb
import h5py, os, numpy as np, time
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm, ticker
from itertools import product
from tqdm import tqdm

pc = 3.0857e18 # cm # parserc

class RunAfterglowPy():
    Z = {'jetType':     grb.jet.TopHat,     # Top-Hat jet
         'specType':    0,                  # Basic Synchrotron Spectrum
         # 'counterjet':  1,
         # 'spread':      0,
         'thetaObs':    0.0,   # Viewing angle in radians
         'E0':          1.0e52, # Isotropic-equivalent energy in erg
         'g0':          1000,
         'thetaCore':   0.2,    # Half-opening angle in radians
         'thetaWing':   0.2,
         'n0':          1e-3,    # circumburst density in cm^{-3}
         'p':           2.2,    # electron energy distribution index
         'epsilon_e':   0.1,    # epsilon_e
         'epsilon_B':   0.01,   # epsilon_B
         'xi_N':        1.0,    # Fraction of electrons accelerated
         'd_L':         3.09e26, # Luminosity distance in cm
         'z':           0.0099}   # redshift
    mapping = {"theta_obs":"thetaObs",
               "Eiso_c":"E0",
               "Gamma0c":"g0",
               "n_ism":"n0",
               "theta_c":"thetaCore",
               "theta_w":"thetaWing",
               "p_fs":"p",
               "eps_e_fs":"epsilon_e",
               "eps_b_fs":"epsilon_B",
               "d_l":"d_L",
               "z":"z"
               }
    times = np.logspace(2, 10, 128)
    freqs = np.logspace(8, 27, 64)
    times_, freqs_ = np.meshgrid(times, freqs, indexing='ij')
    def __init__(self, par_list:list[dict]):
        self.par_list = par_list

    def __call__(self, idx):
        pars = self.par_list[idx]
        Z = copy.deepcopy( self.Z )
        for key, val in pars.items():
            # if not np.isfinite(val):
            #     raise ValueError("Value for {} is not finite".format(key))
            if key in list(self.mapping.keys()):
                Z[self.mapping[key]] = val
        # run the code
        Fnu = grb.fluxDensity(self.times_.flatten(), self.freqs_.flatten(), **Z)
        # reshape the result -> 2D array
        Fnu = np.reshape(Fnu, newshape=self.times_.shape)
        vals = np.array([pars[key] for key in list(pars.keys())])
        return (vals, Fnu)

def get_str_val(v_n, val):
    # if v_n == "theta_obs":
    #     val = "{:.0f}".format(val * 180.0 / np.pi) # rad -> deg
    # elif v_n == "d_l":
    #     val = "{:.1f}".format(val / cgs.pc / 1.e9)
    # else:
    #     val = str(val)
    #
    # return val.replace(".", "")
    if ((v_n == "theta_obs") or (v_n == "theta_c") or (v_n == "theta_w")):
        val = "{:.1f}".format(val / np.pi * 180.) # int(val / np.pi * 180.)
    elif ((v_n == "Eiso_c") or ((v_n == "Eiso_c"))):
        val = np.log10(val)
    elif (v_n == "d_l"):
        val = val / 1.e6 / pc
    else:
        pass
    if len(str(val)) > 7:
        val = "{:.5f}".format(val)
    val = str(val).replace(".", "")
    return val

if __name__ == '__main__':
    run_async = False
    pars = {
        "n_ism":    np.array([1.0, 0.1, 0.01, 0.001]),
        "theta_obs":np.array([0., 15., 45.0, 60., 75., 90.]) * np.pi / 180.0,  # [75*np.pi/180]#
        "Eiso_c":   np.array([1.e50, 1.e51, 1.e52, 1.e53]),
        "Gamma0c":  np.array([100., 300., 600., 1000.]),
        "theta_c":  np.array([5., 10., 15., 20.]) * np.pi / 180.,
        "theta_w":  np.array([5., 10., 15., 20.]) * np.pi / 180.,
        "p_fs":     np.array([2.2, 2.4, 2.6, 2.8]),  # [2.05, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3.0],
        "eps_e_fs": np.array([0.5, 0.1, 0.01, 0.001]),  # [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        "eps_b_fs": np.array([0.5, 0.1, 0.01, 0.001]),  # [0.001, 0.005, 0.01, 0.05, 0.1],
    }

    ranges = [pars[key] for key in pars.keys()]
    pars_cobinations = []
    all_combinations = product(*ranges)
    for combination in all_combinations:
        if "theta_c" in pars and "theta_w" in pars:
            # skip if jet core exceeds jet winds (artifact of permutations)
            if combination[list(pars.keys()).index("theta_c")] > combination[list(pars.keys()).index("theta_w")]:
                continue
        # create a dict with {par:value} for each parameter and value in current permitation
        pars_cobinations.append({par:val for par,val in zip(list(pars.keys()), combination)})
        # create a str containing the par_value for each par and value (used later to label the simulation)
        # result[-1]["name"] = "".join([par.replace("_","")+get_str_val(par,val)+'_'
        #                               for par,val in zip(pars.keys(),combination)])

    start_time = time.perf_counter()

    afgpy = RunAfterglowPy(pars_cobinations)

    if not run_async:
        ncpus = os.cpu_count()
        with Pool(ncpus) as pool:
            results = pool.map(afgpy, range(len(pars_cobinations)))
    else:
        results = []
        processes = os.cpu_count()
        pool = Pool(processes=1)
        jobs = [pool.apply_async(func=afgpy, args=(*argument,)) if isinstance(argument, tuple)
                else pool.apply_async(func=afgpy, args=(argument,)) for argument in range(len(pars_cobinations))]
        pool.close()
        for job in tqdm(jobs):
            results.append(job.get())

    finish_time = time.perf_counter()

    # format the result to [i_feature_set, i_time, i_freq]
    params = np.vstack([res[0] for res in results])
    spectra = np.stack([res[1] for res in results])

    print(f"Program finished in {finish_time-start_time:.2f} seconds. "
          f"Parms={params.shape} spectra={spectra.shape}")

    # save the final result
    outdir = os.getcwd() + '/'
    with h5py.File(outdir+"X_afgpy.h5","w") as f:
        f.create_dataset("X", data=spectra, dtype=np.float32)
        f.create_dataset("times", data=RunAfterglowPy.times)
        f.create_dataset("freqs", data=RunAfterglowPy.freqs)
    with h5py.File(outdir+"Y_afgpy.h5","w") as f:
        f.create_dataset("Y", data=params)
        f.create_dataset("keys", data=np.array(list(pars.keys()),dtype="S"))


    # # plot the time-freq grid
    # fig, ax = plt.subplots()
    # # cs = ax.contourf(times_, freqs_, Fnu, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    # cs = ax.scatter(RunAfterglowPy.times_.flatten(),
    #                 RunAfterglowPy.freqs_.flatten(),
    #                 c=np.log10(spectra[0].flatten()), cmap="viridis", s=0.5)
    # cbar = fig.colorbar(cs)
    # cbar.set_label('log10(Fnu [mJy])')
    # ax.set_xlabel(f'Time [s] #{len(RunAfterglowPy.times)}')
    # ax.set_ylabel(f'Freqs [Hz] #{len(RunAfterglowPy.freqs)}')
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.title("Example Spectra")
    # plt.tight_layout()
    # plt.savefig(outdir+'time_freq_grid.png',dpi=256)
    # plt.close()



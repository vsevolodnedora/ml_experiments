# Creating surrogate afterglow model for afterglowpy

## Preparing the Data

In this part of the project a conditional variational 
autoencoder is applied to the 2D images (spectra) generated with 
afterglowpy. The training data is >20Gb and is not part of the repo. 
The data can be generated using the 
```bash
python3 grid_explore.py 
```
which will iterate over all possible combination of GRB afterglow
model free parameters, generate spectra for a 2D grid of 
observer times and frequencies and save all spectra in 2 HDF5 files, 
'X_afgpy.h5' and 'Y_afgpy.h5' that then directly be used in NN training. 

__NOTE__: It is better to replace afterglowpy with jetsimpy here, as 
it is more similar to PyBlastAfterglow, at least in terms of light curves.  

## Normalizing the Training Data

The key part of using cVAE is to properly scale and normalize the data. 
See Kamile repo for her [kilonovanet](https://github.com/klukosiute/kilonovanet) 
and her [paper](https://arxiv.org/abs/2204.00285) about it. 
Here we first transform fluxes to log10(fluxes) and then do 'minmax' 
normalization. This is done in 
```bash
python3 data.py 
```
The script processes the 'X_afgpy.h5' and 'Y_afgpy.h5' files, cleans and 
normalizes them, and generates a set of metadata files that are used later 
to make inferences on the NN model without the need to store the 
entire training dataset. 

## Training Neural Network

We use a standard Encoder-Decoder architecture for cVAE with 
convolutional layers, ReLu activations and BatchNorm. 
See 'cVAE(nn.Module)' class in 
```bash 
python3 new_train_model.py
```
Note, Kimile in her _kilonovanet_ used linear layers as she 
was trining the model to reproduce 1D images, light curve. 

__NOTE__: it would be better to use a combination of convolutional 
and recurrent layers as light curves are essencially time-series data. 

## Miscellaneous

Other files in the directory are experiments and ideas. 
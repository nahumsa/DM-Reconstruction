# Reconstruction of a Density Matrix using Neural Networks

This repository focus on reconstructing density matrices using measurements. 

For reproducibility's sake, every notebook has a google colab link that you can click on and run the notebook on your browser. *In the future I plan on doing a requirements.txt for pip instalation to run on a local machine.*

## Notebooks

In the notebooks you will find the results for several methods for reconstruction of the density matrices. For example:

- `Reconstruction_MSE.ipynb` : We used the Mean Squared Error of measurements to reconstruct. 

- `Reconstruction DM.ipynb` : We used the [trace distance](https://en.wikipedia.org/wiki/Trace_distance) between the reconstructed density matrix and original density matrix.

- `Reconstruction DM_entropy.ipynb` : We used the relative entropy between the reconstructed density matrix and original density matrix.

## Modules

- Utils:
    - `Dataset.py` : Creating and handling the dataset.
    - `Plotter.py` : Plotting during training.
    - `QutipUtils.py` : Qutip helper functions.
    - `QMeasures.py` : Tensorflow implementation of quantum metrics: trace distance,etc.
    
- Models:
    - `VAE_Keras.py` : Variational Autoencoder implementation on keras.
    - `TVAE.py` : Variational Autoencoder implementation using trace distance as a loss function.

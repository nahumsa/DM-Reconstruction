# Reconstruction of a Density Matrix using Neural Networks

This repository focus on reconstructing density matrices using measurements. 

For reproducibility's sake, every notebook has a google colab link that you can click on and run the notebook on your browser. *In the future I plan on doing a requirements.txt for pip instalation to run on a local machine.*

## Notebooks

In the notebooks you will find the results for several methods for reconstruction of the density matrices. For example:

- `Reconstruction_MSE.ipynb` : We used the Mean Squared Error of measurements to reconstruct. 

- `Reconstruction DM.ipynb` : We used the [trace distance](https://en.wikipedia.org/wiki/Trace_distance) between the reconstructed density matrix and original density matrix.

- `Analysis Results.ipynb` : Used for analysing the results obtained by other notebooks.

- `Autoencoder Benchmark.ipynb` : We use Autoencoders to benchmark our reconstruction results.

- `Autoencoder Benchmark.ipynb` : We use Autoencoders to benchmark our reconstruction results.

- `CreateMeasurements.py` : Python file to create the dataset.



## Modules

- Data: (Data used)
    - `Measurements`: Measurement data with `X_train.txt` and `X_test.txt`.

- Results: (Results obtained)
    - `AE`: Results for the Autoencoder with MSE as a loss function.
    - `TAE`: Results for the Autoencoder with Trace Distance as a loss function.
    - `TVAE`: Results for the Variational Autoencoder with Trace Distance as a loss function.

- Utils: 
    - `Dataset.py` : Creating and handling the dataset.
    - `Plotter.py` : Plotting during training.
    - `QutipUtils.py` : Qutip helper functions.
    - `QMeasures.py` : Tensorflow implementation of quantum metrics: trace distance,etc.
    
- Models: (Deep Learning Models used)
    - `VAE_Keras.py` : Variational Autoencoder implementation on keras.
    - `TVAE.py` : Variational Autoencoder implementation using trace distance as a loss function.
    - `EVAE.py` : Variational Autoencoder implementation using quantum entropy measurements as a loss function. (Not completed)
    - `AE.py` : Autoencoder implementation on keras.
    - `TAE.py` : Autoencoder implementation using trace distance as a loss function.

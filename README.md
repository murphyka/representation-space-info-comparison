# representation-space-info-comparison
Code accompanying "Comparing information content of representation spaces for disentanglement with VAE ensembles" (2024)
by Kieran A Murphy, Sam Dillavou, and Dani S. Bassett.

**TL;DR:** We propose a lightweight methodology for comparing the information content of channels of VAEs.  Given an ensemble of O(50) trained VAEs, we find pieces of information that are repeatedly found across training runs, performing an empirical study of information fragmentation in the context of disentanglement.

Contents:
- `utils.py`: Code to compute pairwise Bhattacharyya distances given a list of Gaussian posterior means and log variances for a sample of datapoints, and then our proposed generalizations for the pairwise normalized mutual information (NMI) and variation of information (VI) values given a list of Bhattacharyya distance matrices
- `fashion_mnist_example.ipynb`: an iPython notebook that reproduces the workflow to analyze structure in ensembles of trained channels
- `ensemble_learning.ipynb`: an iPython notebook that reproduces the ensemble learning example from Sec. 4.4 of the manuscript.

The iPython notebook `fashion_mnist_example.ipynb` will download accessory files (uploaded to [google drive](https://drive.google.com/file/d/1LU5Lcf-wPR9UnOfWNVyXuZRxDQeOzXzX/view?usp=drive_link))
- `trained_fashion_mnist_beta4/`: one $\beta$-VAE model trained on Fashion-MNIST with $\beta=4$,
- `bhats.npy`: a sample of 300 $\times$ 300 Bhattacharyya distance matrices, coming from 25 $\beta$-VAEs trained on Fashion-MNIST with 10 latent dimensions each, so 250 matrices (175MB).
- `utils.py`: The same file in this repository, just provided for convenience.
  
The notebook calculates distinguishability matrices for the channels of the sample model, calculating Bhattacharyya coefficients with a random sample of images from Fashion-MNIST, and then computes the pairwise VI and NMI values from the included Bhattacharyya matrices.
The VI and NMI values are then clustered with sklearn's OPTICS and visualized in the same manner as Fig. 5, reproduced below:

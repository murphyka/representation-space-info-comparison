# representation-space-info-comparison
Code accompanying "Comparing information content of representation spaces for disentanglement with VAE ensembles" (2024) 
by Kieran A Murphy, Sam Dillavou, and Dani S. Bassett [[arxiv](https://arxiv.org/abs/2405.21042)]

**TL;DR:** We propose a lightweight methodology for comparing the information content of channels of VAEs.  Given an ensemble of O(50) trained VAEs, we find pieces of information that are repeatedly found across training runs, allowing an empirical study of information fragmentation in the context of disentanglement.

![Figure from the manuscript that gives a high level overview of the proposed method.](/images/high_level.png)

Contents:
- `utils.py`: Code to compute pairwise Bhattacharyya distances given a list of Gaussian posterior means and log variances for a sample of datapoints, and then our proposed generalizations for the pairwise normalized mutual information (NMI) and variation of information (VI) values given a list of Bhattacharyya distance matrices
- `fashion_mnist_example.ipynb`: an iPython notebook that reproduces the workflow to analyze structure in ensembles of trained channels
- `ensemble_learning.ipynb`: an iPython notebook that reproduces the ensemble learning example from Sec. 4.4 of the manuscript.
- `analyze_locatello19/`: a directory with code to process any of the ensembles of models released with [Challenging Common Assumptions in the Unsupervised Learning of
Disentangled Representations (Locatello et al., 2019)](https://proceedings.mlr.press/v97/locatello19a/locatello19a.pdf)

The iPython notebook `fashion_mnist_example.ipynb` will download accessory files (zipped and uploaded to [google drive](https://drive.google.com/file/d/1LU5Lcf-wPR9UnOfWNVyXuZRxDQeOzXzX/view?usp=drive_link))
- `trained_fashion_mnist_beta4/`: one $\beta$-VAE model trained on Fashion-MNIST with $\beta=4$,
- `bhats.npy`: a sample of 300 $\times$ 300 Bhattacharyya distance matrices, coming from 25 $\beta$-VAEs trained on Fashion-MNIST with 10 latent dimensions each, so 250 matrices (175MB).
- `utils.py`: The same file in this repository, just provided for convenience.
  
The notebook calculates distinguishability matrices for the channels of the sample model, calculating Bhattacharyya coefficients with a random sample of images from Fashion-MNIST, and then computes the pairwise VI and NMI values from the included Bhattacharyya matrices.
The VI and NMI values are then clustered with sklearn's OPTICS and visualized in the same manner as Figs. 4 and 5.  Below, we reproduce the parts of those figures for the MNIST and Fashion-MNIST datasets, along with latent traversals.
![Figure from the manuscript showing structure found in an ensemble of beta VAEs trained on the MNIST dataset.  A block diagonal matrix of the similarities between channels indicates that there are channels repeatedly found throughout the ensemble. Latent traversals from each hot spot visualize the information content.](/images/mnist.png)

![Figure from the manuscript showing structure found in an ensemble of beta VAEs trained on the Fashion MNIST dataset.  A block diagonal matrix of the similarities between channels indicates that there are channels repeatedly found throughout the ensemble. Latent traversals from each hot spot visualize the information content.](/images/fashion_mnist.png)

---
### Full pipeline to analyze the models that were publicly released with Locatello et al. (2019)

While we trained MNIST and Fashion-MNIST ensembles ourselves, all other ensembles used models uploaded with [`disentanglement_lib`](https://github.com/google-research/disentanglement_lib/tree/master) for [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations (Locatello et al. 2019)](https://proceedings.mlr.press/v97/locatello19a.html).  An example is the `cars3d` analysis reproduced below.

![Figure from the manuscript showing structure found in an ensemble of beta VAEs trained on the cars3d dataset.  A block diagonal matrix of the similarities between channels indicates that there are channels repeatedly found throughout the ensemble. Latent traversals from each hot spot visualize the information content.](/images/cars3d.png)

Pipeline:
1. Download the desired models.  For convenience, we've included a shell script that will do it for you, `download_trained_models.sh`.
Just find the start and end model indices (**inclusive**)
[described here in `disentanglement_lib`](https://github.com/google-research/disentanglement_lib/tree/master?tab=readme-ov-file#pretrained-disentanglement_lib-modules).

> **Example calls:**
>
> - `./download_trained_models.sh 0 49`: downloads the 50 `dsprites` $\beta$-VAE models with $\beta=1$.
>
> - `./download_trained_models.sh 7500 7799`: downloads the 300 `smallnorb` FactorVAE models corresponding all six values of $\gamma$.
>
> **Here's a guide to the model indices, where every 300 models spans 6 hyperparameter values for 50 runs each:**
>
> $\beta$-VAE: 0-299
>
> FactorVAE: 300-599
>
> DIP-VAE-I: 600-899
>
> DIP-VAE-II: 900-1199
>
> $\beta$-TCVAE: 1200-1499
>
> Annealed VAE (CCI-VAE): 1500-1799
>
> **Then add the offset for the dataset:**
>
> `dsprites`: 0
>
> `noisy-dsprites`: 1800
>
> `color-dsprites`: 3600
>
> `scream-dsprites`: 5400
>
> `smallnorb`: 7200
>
> `cars3d`: 9000

2. Run the analysis script, `structure_from_trained_models.py`.  First, it prepares a random fingerprint set of images by calling `dataset_helper_fns.py`; for `dsprites` it just uses `tensorflow_datasets`, but for the other datasets you'll need to download them following the instructions in `disentanglement_lib`.
Then the script computes the Bhattacharyya matrices for the model numbers you want to compare, the pairwise NMI and VI matrices, and finally visualizes structure using OPTICS in the same manner as in the manuscript.  For speed, you'll want to try a `fingerprint_size` of 100 or 300 before seeing if 1000 makes any difference.

>**Args:**
>- `fingerprint_size`: This is the number of images to use in the fingerprint.  We used 1000 for the results in the paper, but things are much faster for 100 or 300 and can be sufficient for preliminary analyses.
>- `dataset`: `dsprites`, `smallnorb`, or `cars3d`.  We have not looked at the `dsprites` variants; you'd just need to add code to assemble a fingerprint set of images.
>You can also (of course) add your own dataset; for these simple architectures it just takes one GPU-day to train an ensemble of 50 models.
>- `outdir`: where you want stuff saved.
>- `model_dir`: the directory containing the trained models
>- `model_start`, `model_end`: the range of models (**exclusive**) to include in the analysis.  E.g., 0 to 50 would analyze the 50 $\beta$-VAEs for `dsprites` with $\beta=1$.

    


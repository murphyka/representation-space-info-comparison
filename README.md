# Code accompanying "Comparing the information content of probabilistic representation spaces" (TMLR 2025) 
by Kieran A Murphy, Sam Dillavou, and Dani S. Bassett [[TMLR](https://openreview.net/forum?id=adhsMqURI1)] [[arxiv](https://arxiv.org/abs/2405.21042)]

![Figure from the manuscript showing structure found in an ensemble of beta VAEs trained on the cars3d dataset.  A block diagonal matrix of the similarities between channels indicates that there are channels repeatedly found throughout the ensemble. Latent traversals from each hot spot visualize the information content.](/images/cars3d.png)

TLDR: We compare probabilistic representation spaces by the information they contain about the dataset.  We look at the consistency of information contained in individual channels of VAEs and InfoGANs, in the full latent spaces, and we perform model fusion.

Contents:
- `Similarity_comparison_synthetic.ipynb`: Notebook to reproduce the results from Sec. 4.1
of the manuscript, where different similarity measures are used to compare nine
synthesized one- and two-dimensional representation spaces.
- `Comparing_full_latent_spaces_with_Monte_Carlo.ipynb`: Notebook with code for Sec. 4.3
of the manuscript, comparing the information content of full latent spaces using Monte Carlo estimates.
- `Rep_space_fusion_SO(2).ipynb`: Notebook to reproduce the fusion learning 
example from Sec. 4.4 of the manuscript.

- `utils.py`: Code to for computing the relevant mutual informations -- using Bhattacharyya fingerprints and using Monte Carlo.
- `structure_from_trained_models.py`: Code to compute similarity matrices, as in Fig. 3, using Bhattacharyya fingerprints.
- `save_fingerprint_sample.py`: Code to grab datasets and save a random sample for use when fingerprinting.
- `download_trained_models.sh`: a convenience shell script to download models for analysis.

## Quickest route to channel similarity results:
> `./download_trained_models.sh 250 269`
>
> `structure_from_trained_models.py --model_start 250 --model_end 270`

This will:
1. Download 20 beta-VAE models trained on `dsprites` into `trained_models/`. 
2. Download the `dsprites` dataset from `tensorflow_datasets`.
3. Save a sample of randomly selected images from the dataset to `artifacts/dsprites_fingerprint.npz`.
4. Compute Bhattacharrya matrices for every channel in each of the 20 models, \~10 seconds.
5. Compute all pairwise NMI and VI values for the 20x20 matrix, \~30 seconds.
6. Save a visualization for each NMI and VI in the same format as Fig. 3.  

## Quickest route to comparing full latent spaces:
> `./download_trained_models.sh 7750 7755`

Then execute code in `Comparing_full_latent_spaces_with_Monte_Carlo.ipynb`, to evaluate consistency across FactorVAEs trained on smallnorb (or change to whatever dataset+models you have downloaded).

## Longer explanation of the channel similarity pipeline:
1. Download the desired models. Find the start and end model indices (**inclusive**)
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

2. Run the analysis script, `structure_from_trained_models.py`.  First, it prepares a random fingerprint set of images by calling `save_fingerprint_sample.py`; for `dsprites` it just uses `tensorflow_datasets`, but for the other datasets you'll need to download them following the instructions in `disentanglement_lib`.
Then the script computes the Bhattacharyya matrices for the model numbers you want to compare, the pairwise NMI and VI matrices, and finally visualizes structure using OPTICS in the same manner as in the manuscript.  For speed, you'll want to try a `fingerprint_size` of 100 or 300 before seeing if 1000 makes any difference.

>**Args:**
>- `fingerprint_size`: This is the number of images to use in the fingerprint.  We used 1000 for the results in the paper, but things are much faster for 100 or 300 and can be sufficient for preliminary analyses.
>- `dataset`: `dsprites`, `smallnorb`, or `cars3d`.  We have not looked at the `dsprites` variants; you'd just need to add code to assemble a fingerprint set of images.
>You can also (of course) add your own dataset; for these simple architectures it just takes one GPU-day to train an ensemble of 50 models.
>- `outdir`: where you want stuff saved.
>- `model_dir`: the directory containing the trained models
>- `model_start`, `model_end`: the range of models (**exclusive**) to include in the analysis.  E.g., 0 to 50 would analyze the 50 $\beta$-VAEs for `dsprites` with $\beta=1$.


![Figure from the manuscript showing structure found in an ensemble of beta VAEs trained on the Fashion MNIST dataset.  A block diagonal matrix of the similarities between channels indicates that there are channels repeatedly found throughout the ensemble. Latent traversals from each hot spot visualize the information content.](/images/fashion_mnist.png)

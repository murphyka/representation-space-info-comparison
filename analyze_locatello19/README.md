# Full pipeline to analyze the models that were publicly released with Locatello et al. (2019)

Pipeline:
1. Download the desired models.  We've included a shell script that will do it for you, `download_trained_models.sh`.
Just find the start and end model indices
[described here in `disentanglement_lib`](https://github.com/google-research/disentanglement_lib/tree/master?tab=readme-ov-file#pretrained-disentanglement_lib-modules).

> Example calls:
>
> - `./download_trained_models.sh 0 49`: downloads the 50 `dsprites` $\beta$-VAE models with $\beta=1$.
>
> - `./download_trained_models.sh 7500 7799`: downloads the 300 `smallnorb` FactorVAE models corresponding all six values of $\gamma$.

2. Run the analysis script, `structure_from_trained_models.py`.  First, it prepares a random fingerprint set of images by calling `dataset_helper_fns.py`.
Then it computes the Bhattacharyya matrices for the model numbers you want to compare, the pairwise NMI and VI matrices, and finally assesses structure using OPTICS and visualizes in the same manner as in the manuscript.

>Args:
>- `fingerprint_size`: This is the number of images to use in the fingerprint.  We used 1000 for the results in the paper, but things are much faster for 100 or 300 and can be sufficient for preliminary analyses.
>- `dataset`: `dsprites`, `smallnorb`, or `cars3d`.  We have not looked at the `dsprites` variants; you'd just need to add code to assemble a fingerprint set of images.
>You can also (of course) add your own dataset; for these simple architectures it just takes one GPU-day to train an ensemble of 50 models.
>- `outdir`: where you want stuff saved.
>- `model_dir`: the directory containing the trained models
>- `model_start`, `model_end`: the range of models to include in the analysis.  E.g., 0 to 50 would analyze the 50 $\beta$-VAEs for `dsprites` with $\beta=1$.

    

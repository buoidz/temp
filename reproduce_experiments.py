from code.algorithm.experiment_config.common import ExperimentConfig
from code.algorithm.dataset import DataLoaderORL, DataLoaderYale, DataLoader
from code.algorithm.nmf.lecture_nmf import Lecture_NMF
from code.algorithm.nmf.kl_divergence_nmf import KL_NMF
from code.algorithm.nmf.l21_nmf import L21_NMF
from code.algorithm.nmf.common import NMF
import itertools

noise_options_dict = {
    # "no_noise": [{"p": 0, "r": 0.0}],
    "salt_and_pepper": [
        # {"p": 0.02, "r": 0.2},
        # {"p": 0.02, "r": 0.5},
        # {"p": 0.02, "r": 0.8},
        # {"p": 0.05, "r": 0.2},
        # {"p": 0.05, "r": 0.5},
        # {"p": 0.05, "r": 0.8},
        # {"p": 0.08, "r": 0.2},
        # {"p": 0.08, "r": 0.5},
        {"p": 0.08, "r": 0.8},
    ],
    # "uniform_random": [
    #     {"p": 1, "var": 0.02, "loc": 0},
    #     {"p": 1, "var": 0.04, "loc": 0},
    #     {"p": 1, "var": 0.06, "loc": 0},
    #     {"p": 1, "var": 0.08, "loc": 0},
    #     {"p": 1, "var": 0.10, "loc": 0},
    #     {"p": 1, "var": 0.12, "loc": 0},
    #     {"p": 1, "var": 0.14, "loc": 0},
    # ],
    # "gaussian_noise": [
    #     {"p": 1, "sd": 0.02, "mu": 0},
    #     {"p": 1, "sd": 0.04, "mu": 0},
    #     {"p": 1, "sd": 0.06, "mu": 0},
    #     {"p": 1, "sd": 0.08, "mu": 0},
    #     {"p": 1, "sd": 0.10, "mu": 0},
    #     {"p": 1, "sd": 0.12, "mu": 0},
    #     {"p": 1, "sd": 0.14, "mu": 0},
    # ],
}

datasets: list[DataLoader]
nfm_methods: list[NMF]

# datasets = [DataLoaderORL, DataLoaderYale]
# nfm_methods = [Lecture_NMF, KL_NMF, L21_NMF]
datasets = [DataLoaderYale]
nfm_methods = [Lecture_NMF, KL_NMF]

if __name__ == "__main__":
    for DATASET, NFM in itertools.product(datasets, nfm_methods):
        for noise_method, noise_kargs_list in noise_options_dict.items():
            for i, noise_kwargs in enumerate(noise_kargs_list):
                ds_name = f"{type(DATASET()).__name__.replace('DataLoader', '')}"
                if noise_method == "no_noise":
                    experiment_name = f"{ds_name}_{NFM().method}_no_noise"
                    noise_method = "salt_and_pepper"  # so the code doesn't error
                else:
                    experiment_name = f"{ds_name}_{NFM().method}_{noise_method}_{8}"
                experiment = ExperimentConfig(
                    name=experiment_name,
                    nmf=NFM(),
                    data_loader=DATASET(reduce=3),
                    noise_method=noise_method,
                    noise_kwargs=noise_kwargs,
                    hidden_d=30,
                    seed=4328,
                    proper_folds=True,
                    num_folds=5,
                )
                niter = 2000 if ds_name == "ORL" else 500
                eps = 10e-8 if ds_name == "ORL" else 10e-5
                experiment.run_experiment(niter=niter, eps=eps)

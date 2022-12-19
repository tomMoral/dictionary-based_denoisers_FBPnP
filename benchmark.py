import pandas as pd
import numpy as np
import itertools
import torch

from joblib import Memory

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from pnp_unrolling.models import SynthesisUnrolled
from pnp_unrolling.datasets import create_imagewoof_dataloader
from pnp_unrolling.train import train


PATH_DATA = "/storage/store2/work/bmalezie/imagewoof/"
RESULTS = "benchmark.csv"
COLOR = True
DEVICE = "cuda:3"
mem = Memory(location='./tmp_unrolled_synthesis/', verbose=0)


@mem.cache
def run_test(params):

    try:
        params_model = {
            "n_layers": params["n_layers"],
            "n_components": params["n_components"],
            "kernel_size": params["kernel_size"],
            "lmbd": params["lmbd"],
            "n_channels": 3 if COLOR else 1,
            "device": DEVICE,
            "dtype": torch.float,
            "avg": False,
            "D_shared": params["D_shared"]
        }

        params_dataloader = {
            "path_data": PATH_DATA,
            "sigma_noise": 0.1,
            "device": DEVICE,
            "dtype": torch.float,
            "mini_batch_size": 10,
            "color": COLOR
        }

        unrolled_net = SynthesisUnrolled(**params_model)
        train_dataloader = create_imagewoof_dataloader(
            **params_dataloader,
            train=True,
        )
        test_dataloader = create_imagewoof_dataloader(
            **params_dataloader,
            train=False,
        )

        optimizer = Adam(unrolled_net.parameters(), lr=0.1)
        scheduler = ExponentialLR(optimizer, gamma=0.9)

        train_loss, test_loss = train(
            unrolled_net,
            train_dataloader,
            test_dataloader,
            optimizer,
            scheduler,
            epochs=50,
            max_batch=20
        )

        results = {
            "avg_test_losses": np.mean(test_loss[:-5])
        }

    except (KeyboardInterrupt, SystemExit):
        raise

    except RuntimeError:
        results = {
            "avg_test_losses": np.inf
        }

    return results


if __name__ == "__main__":

    hyperparams = {
        "n_components": [50, 100, 200, 400],
        "kernel_size": [5, 10, 15],
        "lmbd": [1e-5, 1e-3, 1e-1],
        "n_layers": [5, 10, 20],
        "D_shared": [True, False]
    }

    keys, values = zip(*hyperparams.items())
    permuts_params = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    dico_results = {}

    for params in permuts_params:
        print(params)
        results = run_test(params)

        # Storing results
        for key in params.keys():
            if key not in dico_results:
                dico_results[key] = [params[key]]
            else:
                dico_results[key].append(params[key])

        for key in results.keys():
            if key not in dico_results:
                dico_results[key] = [results[key]]
            else:
                dico_results[key].append(results[key])

    results = pd.DataFrame(dico_results)
    results.to_csv(RESULTS)

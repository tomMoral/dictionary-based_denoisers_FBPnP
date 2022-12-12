import pandas as pd
import numpy as np
import itertools

from pnp_unrolling.unrolled_denoiser import BaseUnrolling
from joblib import Memory

DATA_PATH = "/storage/store2/work/bmalezie/imagewoof/"
RESULTS = "benchmark.csv"
mem = Memory(location='./tmp_unrolled_synthesis/', verbose=0)


@mem.cache
def run_test(params):

    try:
        unrolled = BaseUnrolling(
            n_layers=params["n_layers"],
            n_components=params["n_components"],
            kernel_size=params["kernel_size"],
            n_channels=3,
            lmbd=params["lmbd"],
            path_data=DATA_PATH,
            etamax=1e10,
            etamin=1e6,
            iterations=100
        )

        avg_train_losses, avg_test_losses, times = unrolled.fit()

        results = {
            "avg_test_losses": np.mean(avg_test_losses[:-10])
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
        "n_components": [100, 200, 400, 800],
        "kernel_size": [10, 15, 20],
        "lmbd": [1e-5, 1e-3, 1e-1],
        "n_layers": [5, 10, 20]
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

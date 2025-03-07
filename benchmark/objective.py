from benchopt import BaseObjective, safe_import_context
from benchopt.config import get_data_path
from time import perf_counter

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch
    import deepinv as dinv
    from deepinv.loss.metric import PSNR, SSIM
    from deepinv.utils.demo import load_degradation
    from deepinv.optim import optim_builder


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Unrolled Denoiser priors"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tomMoral/pnp_unrolling"

    min_benchmark_version = "1.6.1"

    requirements = ["pip::deepinv"]

    parameters = {
    #     'task': ["denoising", "blur"],
        'sigma': [1e-2, 5e-2, 1e-1]
    }

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.6"

    def run_pnp(self, denoiser, task, lmbd):

        x0 = self.test_dataset[0][0][None]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if task == "denoising":
            physics = dinv.physics.Denoising(sigma=self.sigma)
            restoration = (
                lambda y, *args, **kwargs: (denoiser(y, lmbd), {})
            )  # noqa: E731
        elif task == "blur":
            kernel_torch = load_degradation(
                "Levin09.npy", get_data_path() / "kernels", index=1
            ).unsqueeze(0).unsqueeze(0)
            physics = dinv.physics.BlurFFT(
                img_size=x0.shape,
                filter=kernel_torch,
                noise_model=dinv.physics.GaussianNoise(sigma=self.sigma),
                device=device
            )

            L = physics.compute_norm(
                torch.randn_like(x0, device=device), tol=1e-4
            ).item()
            step_size = 1 / L

            prior = dinv.optim.PnP(denoiser)
            restoration = optim_builder(
                "PGD", prior=prior, data_fidelity=dinv.optim.L2(),
                max_iter=1000,
                params_algo=dict(stepsize=step_size, g_param=lmbd)
            )
            restoration.eval()
        else:
            raise ValueError(f"Unknown task {task}")

        res = []
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_dataset):
                x = x[None, None].to(device)

                y = physics(x)
                # Reset the denoiser's warm-start information if it exists
                if hasattr(denoiser, 'reset'):
                    denoiser.reset()
                t_start = perf_counter()
                x_hat, metrics = restoration(
                    y, physics, compute_metrics=True, x_gt=x
                )
                runtime = perf_counter() - t_start
                if metrics:
                    res.extend([{
                        'task': task, 'lmbd': lmbd, 'id_img': i, 'iter': k,
                        **{k: v for k, v in zip(metrics.keys(), v)}
                    } for k, v in enumerate(zip(*[v[0] for v in metrics.values()]))])

                res.append({
                    'task': task,
                    'lmbd': lmbd,
                    'PSNR': PSNR()(x_hat, x).detach().item(),
                    'SSIM': SSIM()(x_hat, x).detach().item(),
                    'id_img': i,
                    'iter': -1,
                    'runtime': runtime,
                })
        return res

    def set_data(self, train_dataset, test_dataset):
        if not isinstance(train_dataset, torch.utils.data.Dataset):
            train_dataset = torch.utils.data.TensorDataset(train_dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def evaluate_result(self, denoiser):

        results = []
        for task in ["denoising", "blur"]:
            for lmbd in [1e-4, 1e-3, 1e-2, 1e-1]:
                results.extend(self.run_pnp(denoiser, task, lmbd))
        return results

    def get_one_result(self):
        return dict(denoiser=lambda x, _: x)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            train_dataset=self.train_dataset,
        )

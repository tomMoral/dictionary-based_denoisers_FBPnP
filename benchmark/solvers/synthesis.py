from pathlib import Path
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import torch

    from benchmark_utils.noisy_dataset import NoisyImgDataset
    from benchmark_utils.warm_start_denoiser import WarmStartDenoiser

    from pnp_unrolling.unrolled_cdl import UnrolledCDL

UNROLLED_MODELS_DIR = Path(__file__).parent.parent


class Solver(BaseSolver):

    name = "synthesis"

    install_cmd = "conda"
    requirements = [f"pip:-e {UNROLLED_MODELS_DIR}"]

    parameters = {
        "n_epochs": [50],
        "n_layers": [1, 20],
        "step_size_scaling, accelerated": [
            (0.9, True),
            (1.9, False),
        ],
        "init_dual": [True, False],
        "denoiser_n_layers": [1, 20],
        "denoiser_accelerated": [1, 20],
        "random_state": [None],
    }

    sampling_strategy = "run_once"

    def set_objective(self, train_dataset):
        self.train_dataset = train_dataset

        x0 = train_dataset[0][0]
        color = x0.dim() == 3 and x0.shape[0] > 1

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = UnrolledCDL(
            type_unrolling="synthesis",
            n_layers=self.n_layers,
            n_components=50,
            kernel_size=5,
            lmbd=1e-4,
            color=color,
            step_size_scaling=self.step_size_scaling,
            accelerated=self.accelerated,
            init_dual=self.init_dual,
            avg=False,
            rescale=False,
            optimizer="adam",
            lr=1e-3,
            epochs=self.n_epochs,
            device=device,
        )

        noisy_dataset = NoisyImgDataset(
            train_dataset, fixed_noise=True, noise_level=0.05,
            random_state=self.random_state
        )
        pin_memory = device != "cpu"
        self.train_dataloader = torch.utils.data.DataLoader(
            noisy_dataset, batch_size=50, shuffle=True,
            pin_memory=pin_memory, num_workers=5
        )

    def run(self, _):

        self.model.fit(self.train_dataloader)

    def get_result(self):
        denoiser = self.model.unrolled_net.clone(
            n_layers=self.denoiser_n_layers,
            accelerated=self.denoiser_accelerated
        )
        return dict(denoiser=WarmStartDenoiser(denoiser))

from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import torch
    from deepinv.models import DnCNN


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'DnCNN'

    parameters = {
        'pretrained': ["download", None]
    }

    sampling_strategy = 'run_once'

    def set_objective(self, train_dataset):

        x0 = train_dataset[0][0]
        n_channels = 3 if (x0.dim() == 3 and x0.shape[0] > 1) else 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.denoiser = DnCNN(
            pretrained=self.pretrained, device=device,
            in_channels=n_channels, out_channels=n_channels
        )

    def run(self, _):
        pass

    def get_result(self):
        return dict(denoiser=self.denoiser)

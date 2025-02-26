import torch


class WarmStartDenoiser(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
        self.device = denoiser.device

        self.current_state = None

    def reset(self):
        self.current_state = None

    def forward(self, x, _):
        res, self.current_state = self.denoiser.forward(x, self.current_state)
        return res

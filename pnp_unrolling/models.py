import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class SynthesisUnrolled(nn.Module):

    def __init__(self, n_layers, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state):

        super().__init__()

        self.dtype = dtype
        self.device = device
        self.n_components = n_components
        self.kernel_size = kernel_size

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.model = nn.ModuleList(
            [SynthesisLayer(
                n_components,
                kernel_size,
                n_channels,
                lmbd,
                device,
                dtype,
                random_state
            ) for i in range(n_layers)]
        )

        self.parameter = nn.Parameter(
            torch.rand(
                self.shape_params,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.convt = F.conv_transpose2d

    def forward(self, x):

        out = torch.zeros(
            (x.shape[0],
             self.n_components,
             x.shape[2] - self.kernel_size + 1,
             x.shape[3] - self.kernel_size + 1),
            dtype=self.dtype,
            device=self.device
        )

        out_old = out.clone()
        t_old = 1.

        for layer in self.model:
            x, out, t_old, out_old = layer(x, out, t_old, out_old)
        out = self.convt(out, self.parameter)

        return torch.clip(out, 0, 1)


class SynthesisLayer(nn.Module):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state):

        super().__init__()

        self.lmbd = lmbd
        self.n_components = n_components
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.random_state = random_state

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        self.parameter = nn.Parameter(
            torch.rand(
                self.shape_params,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        )

        self.step = 1.

        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_prior = fft.fftn(self.parameter, axis=(2, 3))
        lipschitz = (
            torch.max(
                torch.max(
                    torch.real(fourier_prior * torch.conj(fourier_prior)),
                    dim=3,
                )[0],
                dim=2,
            )[0]
            .sum()
            .item()
        )
        if lipschitz == 0:
            lipschitz = 1.0
        return lipschitz

    def forward(self, x, z, t_old, z_old):

        self.step = 1 / self.compute_lipschitz()

        result1 = self.convt(z, self.parameter)
        result2 = self.conv(
            (result1 - x),
            self.parameter
        )

        out = z - self.step * result2
        thresh = torch.abs(out) - self.step * self.lmbd
        out = torch.sign(out) * F.relu(thresh)

        # FISTA
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
        z = out + ((t_old-1) / t) * (out - z_old)

        return x, z, t, out

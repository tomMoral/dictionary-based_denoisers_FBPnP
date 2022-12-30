import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class UnrolledNet(nn.Module):
    """
    Unrolled network based on Synthesi or Analysis Dictionary Learning

    Parameters
    ----------
    n_layers : int
        Number of iterations/layers
    n_components : int
        Number of atoms/components in the dictionaries
    kernel_size : int
        Size of the atoms/components
    n_channels : int
        Number of channels in images
    lmbd : float
        Regluarization parameter of FISTA
    device : str
        Device for computations
    dtype : type
        Type of tensors
    type_layer : str
        Type of layer, either 'analysis' or 'synthesis'
    random_state : int, optional
        Seed, by default 2147483647
    avg : bool, optional
        Work on normalized images, by default False
    D_shared : bool, optional
        Share dictionaries among layers, by default True
    """

    def __init__(
        self,
        n_layers,
        n_components,
        kernel_size,
        n_channels,
        lmbd,
        device,
        dtype,
        type_layer="synthesis",
        random_state=2147483647,
        avg=False,
        D_shared=True
    ):

        super().__init__()

        self.dtype = dtype
        self.device = device
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.avg = avg

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.parameter = nn.Parameter(
            torch.rand(
                self.shape_params,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device,
            )
        )

        if D_shared:
            D_init = self.parameter

        else:
            D_init = None

        self.type_layer = type_layer

        if self.type_layer == "synthesis":

            self.model = nn.ModuleList(
                [SynthesisLayer(
                    n_components,
                    kernel_size,
                    n_channels,
                    lmbd,
                    device,
                    dtype,
                    random_state,
                    D_shared=D_init
                ) for i in range(n_layers)]
            )

        elif self.type_layer == "analysis":

            self.model = nn.ModuleList(
                [AnalysisLayer(
                    n_components,
                    kernel_size,
                    n_channels,
                    lmbd,
                    device,
                    dtype,
                    random_state,
                    D_shared=D_init
                ) for i in range(n_layers)]
            )

        self.convt = F.conv_transpose2d

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dictionary = fft.fftn(self.parameter, axis=(2, 3))
        lipschitz = (
            torch.max(
                torch.max(
                    torch.real(
                        fourier_dictionary * torch.conj(fourier_dictionary)
                    ),
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

    def forward(self, x):

        if self.avg:
            current_avg = torch.mean(x, axis=(2, 3), keepdim=True)
            current_std = torch.std(x, axis=(2, 3), keepdim=True)
            x_avg = (x - current_avg) / current_std
        else:
            x_avg = x

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

        if self.type_layer == "synthesis":
            for layer in self.model:
                x_avg, out, t_old, out_old = layer(x_avg, out, t_old, out_old)

            out = self.convt(out, self.parameter)    

        elif self.type_layer == "analysis":
            for layer in self.model:
                x_avg, out = layer(x_avg, out)

            step = 1. / self.compute_lipschitz()
            out = x_avg - step * self.convt(out, self.parameter)

        if self.avg:
            out = out * current_std + current_avg

        return torch.clip(out, 0, 1)


class UnrolledLayer(nn.Module):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, D_shared=None):

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

        if D_shared is None:

            self.parameter = nn.Parameter(
                torch.rand(
                    self.shape_params,
                    generator=self.generator,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        else:

            self.parameter = D_shared

        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dictionary = fft.fftn(self.parameter, axis=(2, 3))
        lipschitz = (
            torch.max(
                torch.max(
                    torch.real(
                        fourier_dictionary * torch.conj(fourier_dictionary)
                    ),
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

    def forward(self, x):

        raise NotImplementedError


class SynthesisLayer(UnrolledLayer):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, D_shared=None):

        super().__init__(
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            device,
            dtype,
            random_state,
            D_shared
        )

    def forward(self, x, z, t_old, z_old):

        step = 1. / self.compute_lipschitz()

        result1 = self.convt(z, self.parameter)
        result2 = self.conv(
            (result1 - x),
            self.parameter
        )

        out = z - step * result2
        # thresh = torch.abs(out) - step * self.lmbd
        # out = torch.sign(out) * F.relu(thresh)
        out = out - torch.clip(
            out,
            -step * self.lmbd,
            step * self.lmbd
        )

        # FISTA
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
        z = out + ((t_old-1) / t) * (out - z_old)

        return x, z, t, out


class AnalysisLayer(UnrolledLayer):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, D_shared=None):

        super().__init__(
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            device,
            dtype,
            random_state,
            D_shared
        )

    def forward(self, x, u):

        step = 1. / self.compute_lipschitz()

        result1 = x - step * self.convt(u, self.parameter)
        result2 = u + self.conv(result1, self.parameter)
        out_u = torch.clip(result2, -self.lmbd / step, self.lmbd / step)

        return x, out_u

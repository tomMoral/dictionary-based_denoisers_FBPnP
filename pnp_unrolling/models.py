import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from .dfb_net import DFBNetconst
from .utils import init_params


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
        D_shared=True,
        activation="soft-thresholding",
        init_dual=True
    ):

        super().__init__()

        self.dtype = dtype
        self.device = device
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.avg = avg
        self.lmbd = lmbd

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.init_dual = init_dual

        if type_layer in ["analysis", "synthesis"]:

            self.parameter = init_params(
                self.shape_params,
                self.generator,
                self.dtype,
                self.device,
                type_layer
            )

            self.D_shared = D_shared
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
                    activation=activation,
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
                    activation=activation,
                    D_shared=D_init
                ) for i in range(n_layers)]
            )

        elif self.type_layer == "dfb_net":

            self.model = DFBNetconst(
                device=device,
                dtype=dtype,
                num_of_layers=n_layers,
                channels=n_channels,
                features=n_components,
                padding=0,
                kernel_size=self.kernel_size
            )

        self.convt = F.conv_transpose2d
        self.conv = F.conv2d

    def set_lmbd(self, lmbd):
        if self.type_layer != "dfb_net":
            with torch.no_grad():
                for layer in self.model:
                    layer.lmbd = lmbd
        self.lmbd = lmbd

    def rescale(self):
        """
        Rescale all parameters
        """
        with torch.no_grad():
            if not self.D_shared:
                for layer in self.model:
                    layer.rescale()
            self.parameter /= np.sqrt(self.compute_lipschitz())

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dico = fft.fftn(self.parameter, dim=(1, 2, 3))
        lipschitz = torch.amax(
            torch.real(fourier_dico * torch.conj(fourier_dico)),
            dim=(1, 2, 3)
        ).sum().item()
        if lipschitz == 0:
            lipschitz = 1
        return lipschitz

    def forward(self, x, out=None):

        if self.avg:
            current_avg = torch.mean(x, axis=(2, 3), keepdim=True)
            current_std = torch.std(x, axis=(2, 3), keepdim=True)
            x_avg = (x - current_avg) / current_std
        else:
            x_avg = x

        if self.type_layer == "synthesis":

            if out is None:
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
                x_avg, out, t_old, out_old = layer(x_avg, out, t_old, out_old)

            reconstruction = self.convt(out, self.parameter)

        elif self.type_layer == "analysis":

            if out is None and self.init_dual:
                out = self.conv(x_avg, self.parameter)

            elif out is None:
                out = torch.zeros(
                    (x.shape[0],
                     self.n_components,
                     x.shape[2] - self.kernel_size + 1,
                     x.shape[3] - self.kernel_size + 1),
                    dtype=self.dtype,
                    device=self.device
                )

            for layer in self.model:
                x_avg, out = layer(x_avg, out)

            # step = 1. / self.compute_lipschitz()
            # reconstruction = x_avg - step * self.convt(out, self.parameter)

            reconstruction = x_avg - self.convt(out, self.parameter)

        elif self.type_layer == "dfb_net":

            reconstruction, out = self.model(
                x_avg,
                x_avg,
                self.lmbd,
                u=out,
            )

        if self.avg:
            reconstruction = reconstruction * current_std + current_avg

        return torch.clip(reconstruction, 0, 1), out


class UnrolledLayer(nn.Module):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, activation, type_layer,
                 D_shared=None):

        super().__init__()

        self.lmbd = lmbd
        self.n_components = n_components
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.random_state = random_state
        self.activation = activation

        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        if D_shared is None:

            self.parameter = init_params(
                self.shape_params,
                self.generator,
                self.dtype,
                self.device,
                type_layer
            )

        else:

            self.parameter = D_shared

        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def rescale(self):
        """
        Rescale the parameter
        """
        self.parameter /= np.sqrt(self.compute_lipschitz())

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dico = fft.fftn(self.parameter, dim=(1, 2, 3))
        lipschitz = torch.amax(
            torch.real(fourier_dico * torch.conj(fourier_dico)),
            dim=(1, 2, 3)
        ).sum().item()
        if lipschitz == 0:
            lipschitz = 1
        return lipschitz

    def forward(self, x):

        raise NotImplementedError


class SynthesisLayer(UnrolledLayer):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, activation, D_shared=None):

        super().__init__(
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            device,
            dtype,
            random_state,
            activation,
            "synthesis",
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
        if self.activation == "soft-thresholding":
            out = out - torch.clip(
                out,
                -step * self.lmbd,
                step * self.lmbd
            )
        elif self.activation == "hard-thresholding":
            out = torch.clip(out, -self.lmbd, self.lmbd)

        # FISTA
        t = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
        z = out + ((t_old-1) / t) * (out - z_old)

        return x, z, t, out


class AnalysisLayer(UnrolledLayer):

    def __init__(self, n_components, kernel_size, n_channels, lmbd,
                 device, dtype, random_state, activation, D_shared=None):

        super().__init__(
            n_components,
            kernel_size,
            n_channels,
            lmbd,
            device,
            dtype,
            random_state,
            activation,
            "analysis",
            D_shared
        )

    def forward(self, x, u):

        # step = 1. / self.compute_lipschitz()

        # result1 = x - step * self.convt(u, self.parameter)
        # result2 = u + self.conv(result1, self.parameter)
        # out_u = torch.clip(result2, -self.lmbd / step, self.lmbd / step)

        gamma = 1.8 / self.compute_lipschitz()
        tmp = x - self.convt(u, self.parameter)
        g1 = u + gamma * self.conv(tmp, self.parameter)
        out_u = g1 - gamma * (
            F.relu(g1 / gamma - self.lmbd / gamma)
            - F.relu(- g1 / gamma - self.lmbd / gamma)
        )

        return x, out_u

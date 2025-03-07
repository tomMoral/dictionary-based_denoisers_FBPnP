import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from .dfb_net import DFBNetconst
from .utils import init_params


class UnrolledNet(nn.Module):
    """
    Unrolled network based on Synthesis or Analysis Dictionary Learning

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
    step_size_scaling : float, optional
        Scaling of the step size, by default None, which maps to
        1.0 for synthesis and 1.8 for analysis.
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
        accelerated=True,
        activation="soft-thresholding",
        avg=True,
        D_shared=True,
        step_size_scaling=None,
        init_dual=True,
        random_state=None,
    ):

        super().__init__()

        self.type_layer = type_layer
        self.lmbd = lmbd
        self.n_components = n_components
        self.kernel_size = kernel_size
        self.activation = activation
        self.step_size_scaling = step_size_scaling
        self.init_dual = init_dual
        self.avg = avg
        self.D_shared = D_shared is not False and D_shared is not None

        self.dtype = dtype
        self.device = device
        self.random_state = random_state

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        self._generator = torch.Generator(self.device)
        if random_state is not None:
            self._generator.manual_seed(random_state)

        if self.type_layer in ["analysis", "synthesis"]:
            if self.type_layer == "synthesis":
                Layer = SynthesisLayer
                if step_size_scaling is None:
                    step_size_scaling = 1.0
            elif self.type_layer == "analysis":
                Layer = AnalysisLayer
                if step_size_scaling is None:
                    step_size_scaling = 1.8

            self.W_ = init_params(
                self.shape_params,
                self._generator,
                self.dtype,
                self.device,
                type_layer
            )

            if self.D_shared:
                if D_shared is True:
                    D_init = self.W_
                else:
                    assert D_shared.shape == self.shape_params, (
                        f"D_shared should have shape {self.shape_params}. "
                        f" Got {D_shared.shape}"
                    )
                    D_init = self.W_ = torch.nn.Parameter(D_shared)
            else:
                D_init = None

            self.model = nn.ModuleList(
                [Layer(
                    n_components,
                    kernel_size,
                    n_channels,
                    device,
                    dtype,
                    random_state=random_state,
                    activation=activation,
                    accelerated=accelerated,
                    D_init=D_init,
                    step_size_scaling=step_size_scaling
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

    def clone(
        self,
        n_layers=None,
        step_size_scaling=None,
        accelerated=None,
    ):
        assert self.D_shared, (
            "Cannot clone the model if D_shared is False"
        )

        if n_layers is None:
            n_layers = len(self.model)
        if step_size_scaling is None:
            step_size_scaling = self.step_size_scaling
        if accelerated is None:
            accelerated = self.accelerated

        return UnrolledNet(
            n_layers=n_layers,
            n_components=self.n_components,
            kernel_size=self.kernel_size,
            n_channels=self.shape_params[1],
            lmbd=self.lmbd,
            device=self.device,
            dtype=self.dtype,
            type_layer=self.type_layer,
            accelerated=accelerated,
            activation=self.activation,
            random_state=self.random_state,
            avg=self.avg,
            D_shared=self.W_.detach().clone(),
            step_size_scaling=step_size_scaling,
            init_dual=self.init_dual
        )

    def set_lmbd(self, lmbd):
        self.lmbd = lmbd

    def rescale(self):
        """
        Rescale all parameters
        """
        with torch.no_grad():
            if not self.D_shared:
                for layer in self.model:
                    layer.rescale()
            self.W_ /= np.sqrt(self.compute_lipschitz())

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dico = fft.fftn(self.W_, dim=(1, 2, 3))
        lipschitz = torch.amax(
            torch.real(fourier_dico * torch.conj(fourier_dico)),
            dim=(1, 2, 3)
        ).sum().item()
        if lipschitz == 0:
            lipschitz = 1
        return lipschitz

    def forward(self, z, u0=None):
        if self.avg:
            current_avg = torch.mean(z, axis=(2, 3), keepdim=True)
            current_std = torch.std(z, axis=(2, 3), keepdim=True)
            z = (z - current_avg) / current_std

        if self.type_layer == "synthesis":
            # Init optim variable with warm-start or 0
            if u0 is None:
                if self.init_dual:
                    u = self.conv(z, self.W_)
                else:
                    u = torch.zeros(
                        (z.shape[0],
                         self.n_components,
                         z.shape[2] - self.kernel_size + 1,
                         z.shape[3] - self.kernel_size + 1),
                        dtype=self.dtype,
                        device=self.device
                    )
            else:
                u = u0

            for it, layer in enumerate(self.model):
                z, u, Tsu = layer(
                    z, u, it, self.lmbd
                )

            reconstruction = self.convt(Tsu, self.W_)
            u_current = u

        elif self.type_layer == "analysis":
            if u0 is None:
                if self.init_dual:
                    v = self.conv(z, self.W_)
                else:
                    v = torch.zeros(
                        (z.shape[0],
                         self.n_components,
                         z.shape[2] - self.kernel_size + 1,
                         z.shape[3] - self.kernel_size + 1),
                        dtype=self.dtype,
                        device=self.device
                    )
            else:
                v = u0

            for it, layer in enumerate(self.model):
                z, v, TAv = layer(z, v, it + 1, self.lmbd)

            reconstruction = z - self.convt(TAv, self.W_)
            u_current = v

        if self.avg:
            reconstruction = reconstruction * current_std + current_avg

        return torch.clip(reconstruction, 0, 1), u_current


class UnrolledLayer(nn.Module):

    def __init__(self, n_components, kernel_size, n_channels,
                 device, dtype, random_state, activation,
                 D_init=None, step_size_scaling=1.0, accelerated=True):

        super().__init__()

        self.n_components = n_components
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.random_state = random_state
        self.activation = activation
        self.step_size_scaling = step_size_scaling
        self.accelerated = accelerated

        self._generator = torch.Generator(self.device)
        if random_state is not None:
            self._generator.manual_seed(random_state)

        self.shape_params = (
            self.n_components,
            n_channels,
            self.kernel_size,
            self.kernel_size
        )

        if D_init is None:
            self.W_ = init_params(
                self.shape_params,
                self._generator,
                self.dtype,
                self.device,
                self.type_layer
            )
        else:
            self.W_ = D_init

        self.conv = F.conv2d
        self.convt = F.conv_transpose2d

    def rescale(self):
        """
        Rescale the parameter
        """
        self.W_.data /= np.sqrt(self.compute_lipschitz())

    def compute_lipschitz(self):
        """
        Compute the Lipschitz constant using the FFT.
        """
        fourier_dico = fft.fftn(self.W_, dim=(1, 2, 3))
        lipschitz = torch.amax(
            torch.real(fourier_dico * torch.conj(fourier_dico)),
            dim=(1, 2, 3)
        ).sum().item()
        if lipschitz == 0:
            lipschitz = 1
        return lipschitz

    def forward(self, z, u, it, lmbd=None):
        """Forward pass of the layer.

        It takes in the initial image z and the current estimate u,
        as well as the current iteration number it and the threshold lmbd.
        """
        raise NotImplementedError


class SynthesisLayer(UnrolledLayer):

    type_layer = "synthesis"

    def get_lmbd_max(self, img):
        with torch.no_grad():
            return torch.max(torch.abs(self.conv(img, self.W_)))

    def forward(self, z, u, it, lmbd):

        tau = self.step_size_scaling / self.compute_lipschitz()

        diff = self.convt(u, self.W_) - z
        TSu = u - tau * self.conv(diff, self.W_)
        if self.activation == "soft-thresholding":
            TSu = TSu - torch.clip(TSu, -tau * lmbd, tau * lmbd)
        elif self.activation == "hard-thresholding":
            TSu = TSu * (torch.abs(TSu) >= lmbd)

        # Inertial acceleration
        if self.accelerated:
            alpha_i = (it + 2.1) / 2.1
            u = (1 + alpha_i) * TSu - alpha_i * u
        else:
            u = TSu

        return z, u, TSu


class AnalysisLayer(UnrolledLayer):

    type_layer = "analysis"

    def get_lmbd_max(self, img):
        return 1

    def forward(self, z, v, it, lmbd):

        nu = self.step_size_scaling / self.compute_lipschitz()

        diff = self.convt(v, self.W_) - z
        TAv = v - nu * self.conv(diff, self.W_)
        TAv = torch.clip(TAv, -lmbd, lmbd)

        # Inertial acceleration
        if self.accelerated:
            alpha_i = (it + 2.1) / 2.1
            v = (1 + alpha_i) * TAv - alpha_i * v
        else:
            v = TAv

        return z, v, TAv

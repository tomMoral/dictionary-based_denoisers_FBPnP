import torch
import torch.nn as nn


def init_params(shape, generator, dtype, device, type_init):

    if type_init == "analysis":
        # weights = -1 + 2 * torch.rand(
        #     shape,
        #     generator=generator,
        #     dtype=dtype,
        #     device=device,
        # )
        # weights /= torch.linalg.norm(weights)
        # nn.init.kaiming_normal(
        #     weights,
        #     a=0,
        #     mode='fan_in'
        # )
        weights = nn.Conv2d(
            in_channels=shape[1],
            out_channels=shape[0],
            kernel_size=shape[2],
            padding=0,
            bias=False,
            device=device,
            dtype=dtype
        ).weight.data

    elif type_init == "synthesis":
        # weights = torch.rand(
        #     shape,
        #     generator=generator,
        #     dtype=dtype,
        #     device=device
        # )
        # norm_atoms = torch.norm(
        #     weights,
        #     dim=(2, 3),
        #     keepdim=True
        # )
        # norm_atoms[
        #     torch.nonzero((norm_atoms == 0), as_tuple=False)
        # ] = 1
        # weights /= norm_atoms
        weights = nn.Conv2d(
            in_channels=shape[1],
            out_channels=shape[0],
            kernel_size=shape[2],
            padding=0,
            bias=False,
            device=device,
            dtype=dtype
        ).weight.data

    params = nn.Parameter(weights)

    return params

import torch
import torch.nn as nn


class SoftShk(nn.Module):
    def __init__(self, n_ch=64):
        super(SoftShk, self).__init__()

        self.n_ch = n_ch
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x, l):
        out = self.relu1(x - l) - self.relu2(-x - l)
        return out


class DFBBlock(nn.Module):
    def __init__(self, channels, features, device, dtype, kernel_size=3, padding=1):
        super(DFBBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.conv_t = nn.ConvTranspose2d(
            in_channels=features,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )

        self.dtype = dtype
        self.device = device
        self.conv_t.weight = self.conv.weight
        self.nl = SoftShk(n_ch=channels)
        self.lip = 1e-3

    def forward(self, u_in, x_ref, l):
        # lip2 = self.op_norm2(x_ref.shape)
        lip2 = self.op_norm2((1, x_ref.shape[1], x_ref.shape[2], x_ref.shape[3]))
        gamma = 1.8 / lip2
        tmp = x_ref - self.conv_t(u_in)
        g1 = u_in + gamma * self.conv(tmp)
        p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
        return p1

    def forward_eval(self, u_in, x_ref, l):  # Suggestion: add the eval function as a param in the function
        gamma = 1.8 / self.lip
        tmp = x_ref - self.conv_t(u_in)
        g1 = u_in + gamma * self.conv(tmp)
        p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
        return p1

    def op_norm2(self, im_size):
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            xtmp = torch.randn(*im_size).type(self.dtype).to(self.device)
            xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
            val = 1
            for k in range(max_iter):
                old_val = val
                xtmp = self.conv_t(self.conv(xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
        return val

    def update_lip(self, im_size):
        with torch.no_grad():
            self.lip = self.op_norm2(im_size).item()


class DFBNetconst(nn.Module):
    def __init__(
        self,
        device,
        dtype,
        channels=3,
        features=64,
        num_of_layers=17,
        kernel_size=3,
        padding=1
    ):
        super(DFBNetconst, self).__init__()

        self.in_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=features,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )
        self.out_conv = nn.ConvTranspose2d(
            in_channels=features,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
            device=device,
            dtype=dtype
        )

        self.out_conv.weight = self.in_conv.weight

        self.linlist = nn.ModuleList(
            [DFBBlock(channels=channels,
                      features=features,
                      kernel_size=kernel_size,
                      padding=padding,
                      dtype=dtype,
                      device=device)
             for _ in range(num_of_layers-2)]
        )

    def forward(self, xref, xin, l, u=None):

        if u is None:
            u = self.in_conv(xin)

        for _ in range(len(self.linlist)):
            u = self.linlist[_](u, xref, l)

        out = torch.clamp(xref - self.out_conv(u), min=0, max=1)

        return out, u

    def update_lip(self, im_size):
        for _ in range(len(self.linlist)):
            self.linlist[_].update_lip(im_size)

    def print_lip(self):
        for _ in range(len(self.linlist)):
            print('Layer ', str(_), self.linlist[_].lip)

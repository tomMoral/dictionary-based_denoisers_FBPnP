# %%
import torch
import torch.nn as nn
from external.network_unet import UNetRes

DEVICE = "cuda:0"



# %%

def load_net():
    print("Loading drunet...")
    net = UNetRes(
        in_nc=3 + 1,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode="strideconv",
        upsample_mode="convtranspose"
    )
    net = nn.DataParallel(net, device_ids=[int(DEVICE[-1])])

    filename = 'checkpoint/drunet_color.pth'
    checkpoint = torch.load(filename,
                            map_location=lambda storage,
                            loc: storage)
    try:
        net.module.load_state_dict(checkpoint, strict=True)
    except:
        net.module.load_state_dict(checkpoint.module.state_dict(),
                                   strict=True)
    
    return net


net = load_net()
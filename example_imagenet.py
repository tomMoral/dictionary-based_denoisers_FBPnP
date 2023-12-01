# %%
import numpy as np
import matplotlib.pyplot as plt
import os

from PIL import Image

# %%

path = "/data/parietal/store2/data/ImageNet/train"


files_images = []
cpt_max = 50000
cpt = 0
for root, _, files in os.walk(path): 
    for file in files:
        cpt += 1
        files_images.append(os.path.join(root, file))
    if cpt >= cpt_max:
        break 


# %%
files_images
# %%
img = Image.open(files_images[-89])
# %%

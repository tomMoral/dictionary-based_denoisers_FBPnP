import torch
import os
import torchvision.transforms.functional as TF
from PIL import Image


class ImageWoofDataset(torch.utils.data.Dataset):

    def __init__(self, path, sigma_noise, device, dtype,
                 random_state=2645982315, train=True):

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.sigma_noise = sigma_noise
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)
        self.files = []
        for root, _, files in os.walk(path):
            for file in files:
                if train and "ILSVRC2012" not in file:
                    self.files.append(os.path.join(root, file))
                elif not train and "ILSVRC2012" in file:
                    self.files.append(os.path.join(root, file))

    def __getitem__(self, index):

        # Load image
        img = Image.open(self.files[index])
        tensor_image = TF.to_tensor(img).to(self.device).type(self.dtype)

        if tensor_image.size()[0] == 1:
            tensor_image = tensor_image.repeat((3, 1, 1))

        # Generate noise
        noise = torch.randn(
            tensor_image.size(),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device
        )
        noise *= self.sigma_noise

        return (torch.clip(tensor_image + noise, 0, 1)[:, :160, :160],
                tensor_image[:, :160, :160])

    def __len__(self):

        return len(self.files)

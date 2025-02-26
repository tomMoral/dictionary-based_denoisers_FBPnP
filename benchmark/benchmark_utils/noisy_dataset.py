import torch
from torch.utils.data import Dataset


class ImgDataset(Dataset):

    def __init__(self, dataset):

        super().__init__()
        # Original dataset
        self.dataset = dataset

    def __getitem__(self, index):

        # Get an image and make sure we have a channel dimension
        img = self.dataset[index]
        if isinstance(img, (tuple, list)):
            img = img[0]

        # Make sure we have a channel dimension
        if img.dim() == 2:
            img = img.unsqueeze(0)

        return img

    def __len__(self):

        return len(self.dataset)


class NoisyImgDataset(Dataset):

    def __init__(
            self, dataset,
            fixed_noise=True, noise_level=0.05, noise_range=(0.3, 0.01),
            random_state=None
    ):

        super().__init__()
        # Original dataset
        self.dataset = ImgDataset(dataset)

        # Noise generation parameter
        self.fixed_noise = fixed_noise
        self.noise_level = noise_level
        self.noise_range = noise_range

        x0 = dataset[0][0]
        device, dtype = x0.device, x0.dtype

        self.generator = torch.Generator(device)
        if random_state is not None:
            self.generator.manual_seed(random_state)

        if fixed_noise:
            self.get_noise_scale = lambda: noise_level
        else:
            def _get_noise_scale():
                scale = torch.rand(
                    1,
                    generator=self.generator,
                    dtype=dtype,
                    device=device
                )
                scale *= (noise_range[0] - noise_range[1])
                return scale + noise_range[1]
            self.get_noise_scale = _get_noise_scale

    def __getitem__(self, index):

        # Get an image from the original dataset
        img = self.dataset[index]

        # Generate noise with the right scale
        noise = torch.randn(
            size=img.shape,
            dtype=img.dtype,
            device=img.device,
            generator=self.generator,
        )
        noise *= self.get_noise_scale()

        img_noise = torch.clip(img + noise, 0, 1)

        return img_noise, img

    def __len__(self):

        return len(self.dataset)

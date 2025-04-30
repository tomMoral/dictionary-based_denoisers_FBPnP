import torch
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as Trans
import requests

from PIL import Image


def create_imagewoof_dataloader(
    path_data,
    max_sigma_noise,
    min_sigma_noise,
    device,
    dtype,
    mini_batch_size=10,
    train=True,
    random_state=2147483647,
    color=False,
    download=False,
    fixed_noise=False
):
    """
    Create dataset from ImageWoof

    Parameters
    ----------
    path_data : str
        Path to data
    sigma_noise : float
        Standard deviation of gaussian noise
    device : str
        Device for computations
    dtype : type
        Type of tensors
    mini_batch_size : int, optional
        Size of mini batches, by default 10
    train : bool, optional
        Train or test set, by default True
    random_state : int, optional
        Seed, by default 2147483647
    color : bool, optional
        Images in color, by default False

    Returns
    -------
    torch.utils.data.DataLoader
        Torch DataLoader
    """
    if download:
        path_data = download_imagewoof(path_data)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return torch.utils.data.DataLoader(
        ImageWoofDataset(
            path_data,
            max_sigma_noise,
            min_sigma_noise,
            device=device,
            dtype=dtype,
            train=train,
            random_state=random_state,
            color=color,
            fixed_noise=fixed_noise
        ),
        batch_size=mini_batch_size,
        shuffle=True,
        generator=generator
    )


def download_imagewoof(path):
    """
    Download data from imagewoof

    Parameters
    ----------
    path : str
        Path where to write data

    Returns
    -------
    str
        Full path to data
    """
    path = os.path.join(path, "imagewoof")
    filename_tar = "imagewoof.tgz"

    try:
        os.makedirs(path)
    except OSError:
        raise Exception(
            f"{path} already exists. "
            "Please remove the folder or put 'download' to False"
            ) from FileExistsError

    full_path = os.path.abspath(path)
    path_tar = os.path.join(full_path, filename_tar)

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
    print("Downloading data...")
    response = requests.get(url)

    with open(path_tar, "wb") as filename:
        filename.write(response.content)

    print("Done")
    print("Extracting data...")

    os.system(f"tar -xzf {path_tar} --directory {full_path}")

    print("Done")
    print("Cleaning folder...")

    os.system(f"rm {path_tar}")
    os.system(
        f"rm {os.path.join(full_path, 'imagewoof2-160/noisy_imagewoof.csv')}"
    )
    print("Done")

    return full_path


class ImageWoofDataset(torch.utils.data.Dataset):

    def __init__(self, path, max_sigma_noise, min_sigma_noise,
                 device, dtype, random_state=2645982315, train=True,
                 color=True, fixed_noise=False):

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.max_sigma_noise = max_sigma_noise
        self.min_sigma_noise = min_sigma_noise
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)
        self.fixed_noise = fixed_noise
        self.color = color
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
        if not self.color:
            img = img.convert("L")
        tensor_image = TF.to_tensor(img).to(self.device).type(self.dtype)

        if self.color and tensor_image.size()[0] == 1:
            tensor_image = tensor_image.repeat((3, 1, 1))

        # Generate noise
        noise = torch.randn(
            tensor_image.size(),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device
        )
        if not self.fixed_noise:
            scale = torch.rand(
                1,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device
            )
        else:
            scale = 1.
        max_diff = (self.max_sigma_noise - self.min_sigma_noise)
        noise *= scale * max_diff + self.min_sigma_noise

        img_noise = torch.clip(tensor_image + noise, 0, 1)[:, :160, :160]
        img_clip = tensor_image[:, :160, :160]

        return img_noise, img_clip

    def __len__(self):

        return len(self.files)


def create_imagenet_dataloader(
    path_data,
    max_sigma_noise,
    min_sigma_noise,
    device,
    dtype,
    mini_batch_size=10,
    train=True,
    random_state=2147483647,
    color=False,
    download=False,
    fixed_noise=False,
    crop=True
):
    """
    Create dataset from ImageNet

    Parameters
    ----------
    path_data : str
        Path to data
    sigma_noise : float
        Standard deviation of gaussian noise
    device : str
        Device for computations
    dtype : type
        Type of tensors
    mini_batch_size : int, optional
        Size of mini batches, by default 10
    train : bool, optional
        Train or test set, by default True
    random_state : int, optional
        Seed, by default 2147483647
    color : bool, optional
        Images in color, by default False

    Returns
    -------
    torch.utils.data.DataLoader
        Torch DataLoader
    """

    generator = torch.Generator()
    generator.manual_seed(random_state)
    return torch.utils.data.DataLoader(
        ImageNetDataset(
            path_data,
            max_sigma_noise,
            min_sigma_noise,
            device=device,
            dtype=dtype,
            train=train,
            random_state=random_state,
            color=color,
            fixed_noise=fixed_noise,
            crop=crop
        ),
        batch_size=mini_batch_size,
        shuffle=True,
        generator=generator
    )


class ImageNetDataset(torch.utils.data.Dataset):

    def __init__(self, path, max_sigma_noise, min_sigma_noise,
                 device, dtype, random_state=2645982315, train=True,
                 color=True, fixed_noise=False, n_max_train=10000,
                 n_max_test=1000, crop_size=150, crop=True):

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.max_sigma_noise = max_sigma_noise
        self.min_sigma_noise = min_sigma_noise
        self.generator = torch.Generator(self.device)
        self.generator.manual_seed(random_state)
        self.fixed_noise = fixed_noise
        self.color = color
        self.files = []
        if crop:
            self.crop_transform = Trans.RandomCrop(
                crop_size,
                pad_if_needed=True,
                padding_mode="symmetric"
            )
        else:
            self.crop_transform = lambda x: x
        if train:
            path = os.path.join(path, "train")
        else:
            path = os.path.join(path, "val")
        for root, _, files in os.walk(path):
            for file in files:
                self.files.append(os.path.join(root, file))

    def __getitem__(self, index):

        # Load image
        img = Image.open(self.files[index])
        if not self.color:
            img = img.convert("L")
        tensor_image = TF.to_tensor(img).to(self.device).type(self.dtype)

        if self.color and tensor_image.size()[0] == 1:
            tensor_image = tensor_image.repeat((3, 1, 1))

        new_img = self.crop_transform(tensor_image)

        # Generate noise
        noise = torch.randn(
            new_img.size(),
            generator=self.generator,
            dtype=self.dtype,
            device=self.device
        )
        if not self.fixed_noise:
            scale = torch.rand(
                1,
                generator=self.generator,
                dtype=self.dtype,
                device=self.device
            )
        else:
            scale = 1.
        max_diff = (self.max_sigma_noise - self.min_sigma_noise)
        noise *= scale * max_diff + self.min_sigma_noise

        img_noise = torch.clip(new_img + noise, 0, 1)

        return img_noise, new_img

    def __len__(self):

        return len(self.files)

from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import torch
    import numpy as np
    import torchvision.transforms as transforms

    from deepinv.utils.demo import load_dataset


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    name = "CBSD68"

    parameters = {
        "n_images": [None],
        "seed": [None]
    }

    def get_data(self):

        img_size = 128 if torch.cuda.is_available() else 32
        transform = transforms.Compose([
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            lambda x: x[0]
        ])
        data_dir = get_data_path()
        train_dataset = load_dataset(
            "CBSD500", data_dir=data_dir, transform=transform
        )
        test_dataset = load_dataset(
            "CBSD68", data_dir=data_dir, transform=transform
        )
        if self.n_images is not None:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(test_dataset))[:self.n_images]
            test_dataset = torch.utils.data.Subset(test_dataset, indices=idx)

        return dict(train_dataset=train_dataset, test_dataset=test_dataset)

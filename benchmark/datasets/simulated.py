from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import torch


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    name = "Simulated"

    parameters = {
        'n_images': [1],
    }

    def get_data(self):

        images = torch.ones(self.n_images, 3, 32, 32) * .5

        return dict(images=images)

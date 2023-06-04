from torch.utils.data import Dataset
import os

# import kornia
import kornia.augmentation as KA
import torch
import xarray as xr


class TestDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        z500 = xr.open_mfdataset(
            os.path.join(root_dir, "geopotential_500/*.nc"),
            combine="by_coords",
            chunks=100,
        ).z

        zmin = z500.min().compute()
        zmax = z500.max().compute()
        self.normalized_z500 = (z500 - zmin) / (zmax - zmin)

        # set up transforms
        if self.transforms is not None:
            self.input_T = KA.container.AugmentationSequential(
                *self.transforms, data_keys=["input"], same_on_batch=False
            )

    def __len__(self):
        return len(self.normalized_z500.time)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.normalized_z500.isel(time=idx).values[None, ...]

        if self.transforms is not None:
            image = self.input_T(image)[0]

        return torch.tensor(image)


def single_torch_file_from_dataset(root_dir):
    z500 = xr.open_mfdataset(
        os.path.join(root_dir, "geopotential_500/*.nc"),
        combine="by_coords",
    ).z
    zmin = z500.min()
    zmax = z500.max()
    normalized_z500 = torch.tensor(((z500 - zmin) / (zmax - zmin)).values)[
        :, None, ...
    ]
    torch.save(
        normalized_z500,
        os.path.join(root_dir, "complete_dataset.pt"),
    )


class SingleDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        self.data = torch.load(os.path.join(root_dir, "complete_dataset.pt"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.data[idx]

        if self.transforms is not None:
            image = self.input_T(image)[0]

        return torch.tensor(image)

import os

from torch.utils.data import DataLoader
from rawdiffusion.datasets.transforms import ImageTransforms
from .raw_image_dataset import RAWImageDataset
from .google_dataset import GoogleDataset
from rawdiffusion.datasets.camera import get_camera


def create_dataset(
    *,
    camera_name,
    data_dir,
    file_list,
    batch_size,
    seed,
    is_train=True,
    transform=True,
    permutate_once=False,
    resample_dataset_size=None,
    min_mode="black_level",
    patch_size=256,
    max_items=None,
    num_workers=8,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    dataset_folder_name = os.path.basename(os.path.normpath(data_dir)).lower()
    dataset_name = dataset_folder_name.split("_")[0]
    if transform:
        transforms = ImageTransforms(
            patch_size=patch_size,
            is_train=is_train,
        )
    else:
        transforms = None

    dataset = GoogleDataset(is_train=is_train, transforms=transforms)

    if permutate_once:
        from .dataset_wrapper import PermutedDataset

        dataset = PermutedDataset(dataset, seed=123)

    if resample_dataset_size is not None:
        from .dataset_wrapper import RandomSampleDataset

        dataset = RandomSampleDataset(dataset, n=resample_dataset_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        drop_last=is_train,
    )
    return loader

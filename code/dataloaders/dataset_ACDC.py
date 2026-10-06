import torch
import random
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import os


class ACDC_dataset(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split not in ["train", "valid", "test"]:
            raise ValueError(
                f"Unsupported split: {self.split}. "
                f"Expected 'train', 'valid', or 'test'."
            )

        list_path = os.path.join(self._base_dir, f"{self.split}.txt")

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Cannot find list file: {list_path}")
        with open(list_path, "r") as f:
            self.sample_list = [item.replace("\n", "") for item in f.readlines()]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            data = np.load(self._base_dir + "/train/{}".format(case))
        elif self.split == "valid":
            data = np.load(self._base_dir + "/valid/{}".format(case))
        else:
            data = np.load(self._base_dir + "/test/{}".format(case))

        image = data["img"]
        label = data["label"]
        sample = {"image": image, "label": label}
        if self.transform is not None and self.split == "train":
            sample = self.transform(sample)
        sample["case_name"] = case
        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32)).long()
        sample = {"image": image, "label": label}
        return sample




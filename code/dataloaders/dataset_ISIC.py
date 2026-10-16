import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy import ndimage


class ISIC2018_dataset(Dataset):
    def __init__(self, base_dir, split="train", img_size=256):
        self.base_dir = base_dir
        self.split = split
        self.img_size = img_size
        self.train = split == "train"

        self.img_dir = os.path.join(
            base_dir, "ISIC2018_Task1-2_Training_Input"
        )
        self.mask_dir = os.path.join(
            base_dir, "ISIC2018_Task1_Training_GroundTruth"
        )

        img_list = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

        if split == "train":
            img_list = img_list[:1815]
        elif split == "val":
            img_list = img_list[1815:1815 + 259]
        elif split == "test":
            img_list = img_list[1815 + 259:]
        else:
            raise ValueError("split must be train / val / test")

        self.img_list = img_list
        self.case_names = []

        data = np.zeros([len(img_list), img_size, img_size, 3], dtype=np.float32)
        mask = np.zeros([len(img_list), img_size, img_size], dtype=np.int64)

        for idx, img_path in enumerate(img_list):
            img_name = os.path.basename(img_path)[:-4]
            self.case_names.append(img_name)

            image = Image.open(img_path).convert("RGB")
            image = image.resize((img_size, img_size), resample=Image.BILINEAR)
            image = np.asarray(image, dtype=np.float32) / 255.0
            data[idx] = image

            mask_path = os.path.join(self.mask_dir,img_name + "_segmentation.png")
            label = Image.open(mask_path).convert("L")
            label = label.resize((img_size, img_size), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.uint8)
            label = (label > 0).astype(np.int64)
            mask[idx] = label

        self.data = data
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        seg = self.mask[idx]

        if self.train:
            if random.random() > 0.5:
                img, seg = self.random_rot_flip(img, seg)

            if random.random() > 0.5:
                img, seg = self.random_rotate(img, seg)

        img = torch.from_numpy(img.copy()).float()
        seg = torch.from_numpy(seg.copy()).long()
        img = img.permute(2, 0, 1)
        sample = {
            "image": img,
            "label": seg,
            "case_name": self.case_names[idx]
        }

        return sample

    def random_rot_flip(self, image, label):
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)

        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return image, label

    def random_rotate(self, image, label):
        angle = np.random.randint(20, 80)
        image = ndimage.rotate(image, angle, order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)

        return image, label
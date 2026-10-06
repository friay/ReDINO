import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, use_strong_aug=True):
        self.output_size = output_size
        self.use_strong_aug = use_strong_aug

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
            )
        ], random_order=True)

    def strong_augment(self, image, label):
        image = image.astype(np.float32)
        label = label.astype(np.int32)
        image_3d = image[..., None]
        aug_det = self.img_aug.to_deterministic()
        image_aug = aug_det.augment_image(image_3d)
        segmap = SegmentationMapsOnImage(label, shape=image_3d.shape)
        label_aug = aug_det.augment_segmentation_maps([segmap])[0].get_arr()
        if label_aug.ndim == 3:
            label_aug = label_aug[..., 0]
        image_aug = image_aug[..., 0]
        image_aug = np.clip(image_aug, 0.0, 1.0)

        return image_aug.astype(np.float32), label_aug.astype(np.int64)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.use_strong_aug:
            image, label = self.strong_augment(image, label)
        else:
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int64))
        sample = {'image': image,'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = base_dir
        self.sample_list = open(os.path.join(base_dir, self.split+'.txt')).readlines()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, "train_npz", slice_name+'.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/test_vol_h5" + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample

# -*- coding: utf-8 -*-
# @Time    : 2021/6/19 11:30 上午
# @Author  : Haonan Wang
# @File    : Load_Dataset.py
# @Software: PyCharm
import numpy as np
import torch
import random
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Callable
import os
import cv2
from scipy import ndimage
import h5py

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
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() < 0.5:
            image, label = random_rotate(image, label)

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

class ValGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = F.to_pil_image(image), F.to_pil_image(label)
        x, y = image.size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = F.to_tensor(image)
        label = to_long_tensor(label)
        sample = {'image': image, 'label': label}
        return sample

def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()

def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images

class ImageToImage2D(Dataset):
    """
    Reads the images and applies the augmentation transform on them.
    Usage:
        1. If used without the unet.model.Model wrapper, an instance of this object should be passed to
           torch.utils.data.DataLoader. Iterating through this returns the tuple of image, mask and image
           filename.
        2. With unet.model.Model wrapper, an instance of this object should be passed as train or validation
           datasets.

    Args:
        dataset_path: path to the dataset. Structure of the dataset should be:
            dataset_path
              |-- images
                  |-- img001.png
                  |-- img002.png
                  |-- ...
              |-- masks
                  |-- img001.png
                  |-- img002.png
                  |-- ...

        joint_transform: augmentation transform, an instance of JointTransform2D. If bool(joint_transform)
            evaluates to False, torchvision.transforms.ToTensor will be used on both image and mask.
        one_hot_mask: bool, if True, returns the mask in one-hot encoded form.
    """

    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False, image_size: int =512) -> None:
        self.dataset_path = dataset_path
        self.image_size = image_size

        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        # print(self.input_path, self.output_path) #datasets/ISIC2017/train/labelcol

        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):

        image_filename = self.images_list[idx]
        #print(image_filename[: -3])
        # read image
        # print(os.path.join(self.input_path, image_filename))
        # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
        # print(os.path.join(self.input_path, image_filename))
        image = cv2.imread(os.path.join(self.input_path, image_filename))
        # print("img",image_filename)
        # print("1",image.shape)
        image = cv2.resize(image,(self.image_size,self.image_size))
        # print(np.max(image), np.min(image))
        # print("2",image.shape)
        # read mask image
        # print(os.path.join(self.output_path, image_filename[: -4] + "_segmentation.png"))
        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_segmentation.png"),0)

        # print(np.max(mask), np.min(mask))
        mask = cv2.resize(mask,(self.image_size,self.image_size))
        # print(np.max(mask), np.min(mask))
        mask[mask<=0] = 0
        # (mask == 35).astype(int)
        mask[mask>0] = 1
        # print("11111",np.max(mask), np.min(mask))

        # correct dimensions if needed
        image, mask = correct_dims(image, mask)
        # image, mask = F.to_pil_image(image), F.to_pil_image(mask)
        # print("11",image.shape)
        # print("22",mask.shape)
        sample = {'image': image, 'label': mask}

        if self.joint_transform:
            sample = self.joint_transform(sample)
        # sample = {'image': image, 'label': mask}
        # print("2222",np.max(mask), np.min(mask))

        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)
        # mask = np.swapaxes(mask,2,0)
        # print(image.shape)
        # print("mask",mask)
        # mask = np.transpose(mask,(2,0,1))
        # image = np.transpose(image,(2,0,1))
        # print(image.shape)
        # print(mask.shape)
        # print(sample['image'].shape)

        return sample, image_filename

# class ImageToImage2D_kfold(Dataset):
#
#     def __init__(self, dataset_path: str,
#                  joint_transform: Callable = None,
#                  one_hot_mask: int = False,
#                  image_size: int =224,
#                  filelists = None,
#                  task_name = None,
#                  split = 'train') -> None:
#         self.dataset_path = dataset_path
#         self.image_size = image_size
#         if task_name == "Synapse":
#             self.input_path = dataset_path
#             self.output_path = None
#         else:
#             self.input_path = os.path.join(dataset_path, 'img')
#             self.output_path = os.path.join(dataset_path, 'labelcol')
#
#         self.images_list = os.listdir(self.input_path)
#         self.one_hot_mask = one_hot_mask
#         self.task_name = task_name
#         self.split = split
#         # print(self.dataset_path[11:15])
#         # print(len(self.images_list))
#         self.images_list = [item for item in self.images_list if item in filelists]
#         # print("img",len(self.images_list))
#
#         if joint_transform:
#             self.joint_transform = joint_transform
#         else:
#             to_tensor = T.ToTensor()
#             self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))
#
#     def __len__(self):
#         return len(self.images_list)
#
#     def __getitem__(self, idx):
#
#         image_filename = self.images_list[idx]
#         #print(image_filename[: -3])
#         # read image
#         # print(os.path.join(self.input_path, image_filename))
#         # print(os.path.join(self.output_path, image_filename[: -3] + "png"))
#         # print(os.path.join(self.input_path, image_filename))
#
#         # print("img",image_filename)
#         # print("1",image.shape)
#
#         # print(np.max(image), np.min(image))
#         # print("2",image.shape)
#         # read mask image
#         if self.task_name == "ISIC2018T1" :
#             image = cv2.imread(os.path.join(self.input_path, image_filename))
#             mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"), 0)
#         elif self.task_name == "ISIC" or self.task_name == "ISIC18":
#             image = cv2.imread(os.path.join(self.input_path, image_filename))
#             mask = cv2.imread(os.path.join(self.output_path, image_filename[: -4] + "_segmentation.png"),0)
#         elif self.task_name == "DR_MA":
#             image = cv2.imread(os.path.join(self.input_path, image_filename))
#             mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "tif"),0)
#         elif self.task_name == "Synapse":
#             if self.split == "train":
#                 data_path = os.path.join(self.input_path, image_filename)
#                 data = np.load(data_path)
#                 image, mask = data['image'], data['label']
#             else:
#                 filepath = self.input_path + image_filename
#                 # print(filepath)
#                 data = h5py.File(filepath)
#                 image, mask = data['image'][:], data['label'][:]
#                 # print(image.shape)
#         else:
#             image = cv2.imread(os.path.join(self.input_path, image_filename))
#             mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "png"),0)
#
#         if self.task_name == "Synapse":
#             if self.split=='train':
#                 image = cv2.resize(image,(self.image_size,self.image_size))
#                 mask = cv2.resize(mask,(self.image_size,self.image_size))
#             else:
#                 self.joint_transform = None
#         else:
#             image = cv2.resize(image,(self.image_size,self.image_size))
#             mask = cv2.resize(mask,(self.image_size,self.image_size))
#             mask[mask<=0] = 0
#             mask[mask>0] = 1
#             image, mask = correct_dims(image, mask)
#
#         sample = {'image': image, 'label': mask}
#
#         if self.joint_transform:
#             sample = self.joint_transform(sample)
#
#
#         return sample, image_filename

import logging
import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, edges_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.edges_dir = edges_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.imgs_dir)
                    if not file.startswith(".")]
        logging.info('Creating dataset with {} examples'.format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC2CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + ".png")
        mask_file = glob(self.masks_dir + idx + ".png")
        edge_file = glob(self.edges_dir + idx + "png")

        assert len(img_file) == 1, \
            'Either no mask or multiple masks found for the ID {}: {}'.format(idx, img_file)
        assert len(mask_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, mask_file)
        assert len(edge_file) == 1, \
            'Either no image or multiple images found for the ID {}: {}'.format(idx, edge_file)
        img = cv2.imread(img_file[0])
        mask = cv2.imread(mask_file[0])
        edge = cv2.imread(edge_file[0])

        assert img.size == mask.size and mask.size == edge.size, \
            'Image, mask and edge {} should be the same size, ' \
            'but are {}, {} and {}'.format(idx, img.size, mask.size, edge.size)

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        edge = self.preprocess(edge, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'edge': torch.from_numpy(edge).type(torch.FloatTensor)
        }


class MoNuSegTrainingDataset(BasicDataset):
    def __init__(self, imgs_dir="./data/MoNuSegTrainingData/Patch/Imgs",
                 masks_dir="./data/MoNuSegTrainingData/Patch/Masks",
                 edges_dir="./data/MoNuSegTrainingData/Patch/Edges", scale=1):
        super().__init__(imgs_dir, masks_dir, edges_dir, scale)


class MoNuSegTestDataset(BasicDataset):
    def __init__(self, imgs_dir="./data/MoNuSegTestData/Patch/Imgs",
                 masks_dir="./data/MoNuSegTestData/Patch/Masks",
                 edges_dir="./data/MoNuSegTestData/Patch/Edges", scale=1):
        super().__init__(imgs_dir, masks_dir, edges_dir, scale)

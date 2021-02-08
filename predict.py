"""Predict and output results"""
import argparse
import logging
import os

import cv2
import numba as nb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from parallelunet import ParallelUNet
from utils import MoNuSegTestDataset

os.environ['CUDA_VISIBLE_DIVICES'] = '0, 1'


@nb.jit(nopython=True)
def convert(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] >= 0.5:
                img[i, j] = 255
            else:
                img[i, j] = 0
    return img


def get_args():
    parser = argparse.ArgumentParser(
        description="Predict masks on trained network (results will be saved under './predictions' dir)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default="",
                        help='Load net from a .pth file')
    parser.add_argument('-o', '--output', dest='output', type=str, default="", help='Name of output dir')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    args = get_args()
    if args.load == "" or args.output == "":
        raise ValueError("load file or output dir name should be specified")

    pth = str(args.load)
    output = str(args.output)

    test_dataset = MoNuSegTestDataset()
    test_loader = DataLoader(test_dataset, num_workers=1, pin_memory=True)

    net = ParallelUNet(3, 1)
    net = nn.DataParallel(net, device_ids=[0, 1])
    net.load_state_dict(torch.load(pth))
    net.cuda()
    net.eval()
    logging.info("Loading weight from {}".format(pth))

    if not os.path.exists("./predictions"):
        os.mkdir("./predictions")
    if not os.path.exists("./predictions/{}".format(output)):
        os.mkdir("./predictions/{}".format(output))
    if not os.path.exists("./predictions/{}/true".format(output)):
        os.mkdir("./predictions/{}/true".format(output))
    if not os.path.exists("./predictions/{}/pred".format(output)):
        os.mkdir("./predictions/{}/pred".format(output))
    logging.info("Output Dir: './predictions/{}'".format(output))

    epoch = 0
    for batch in tqdm(test_loader):
        imgs, true_masks, true_edges = batch['image'], batch['mask'], batch['edge']
        imgs = imgs.cuda()
        true_masks = true_masks.cuda()
        true_edges = true_edges.cuda()

        with torch.no_grad():
            masks_pred, edges_pred = net(imgs)

        cv2.imwrite("./predictions/{}/true/{}.png".format(output, epoch), convert(true_masks.cpu().numpy().squeeze()))
        cv2.imwrite("./predictions/{}/pred/{}.png".format(output, epoch), convert(masks_pred.cpu().numpy().squeeze()))

        epoch += 1

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autoencoder import AutoEncoder
from eval_autoencoder import eval_net
from unet import UNet
from utils import MoNuSegTrainingDataset, MoNuSegTestDataset

os.environ['CUDA_VISIBLE_DIVICES'] = "0, 1, 2"

DIR_CHECKPOINTS = './checkpoints/autoencoder/'


def train_net(netx,
              nety,
              netz,
              epochs=5,
              batch_size=1,
              lr=3e-4,
              save_cp=True,
              img_scale=0.5,
              val_percent=0.1,
              load_args=False,
              **kwargs):
    n_classes = netx.n_classes
    n_channels = netx.n_channels

    writer = SummaryWriter(comment='AUTOENCODER_LR_{}_BS_{}_SCALE_{}'.format(lr, batch_size, img_scale))

    netx = nn.DataParallel(netx, device_ids=[0, 1])
    nety = nn.DataParallel(nety, device_ids=[0, 1])
    netz = nn.DataParallel(netz, device_ids=[0, 1])

    netx.load_state_dict(torch.load("./checkpoints/unet_mask/UNet_CP_epoch29.pth"))
    netx.cuda()

    nety.load_state_dict(torch.load("./checkpoints/unet/UNet_CP_epoch3.pth"))
    nety.cuda()

    if load_args:
        netz.load_state_dict(torch.load(kwargs["pth_file"]))
        logging.info('Model loaded from {}'.format(kwargs["pth_file"]))
    else:
        for m in netz.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    netz.cuda()

    train_dataset = MoNuSegTrainingDataset()
    test_dataset = MoNuSegTestDataset()
    n_test = len(test_dataset)
    n_val = int(len(train_dataset) * val_percent)
    n_train = len(train_dataset) - n_val
    train, val = random_split(train_dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=9, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True,
                            num_workers=9, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=9, pin_memory=True)

    global_step = 0

    logging.info("""Starting training:
        Epochs:             {}
        Batch size:         {}
        Learning rate:      {}
        Training size:      {}
        Validation size:    {}
        Test size:          {}
        Checkpoints:        {}
        Device(s):          {}
        Image scaling:      {}
    """.format(epochs, batch_size, lr, n_train, n_val, n_test, save_cp, device.type, img_scale))

    optimizer = optim.Adam(netz.parameters(), lr=lr, weight_decay=1e-8)

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        netx.eval()
        nety.eval()
        netz.train()

        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == n_channels, \
                    'Network has been defined with {} input channels, ' \
                    'but loaded images have {} channels. Please check that ' \
                    'the images are loaded correctly.'.format(n_channels, imgs.shape[1])

                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

                with torch.no_grad():
                    input1 = netx(imgs)
                    input2 = nety(imgs)
                input = torch.cat([input1, input2], dim=1)
                masks_pred = netz(input)
                loss = criterion(masks_pred, true_masks)
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{"loss {batch}": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(netz.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

        for tag, value in netz.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
        val_score = eval_net(netx, nety, netz, val_loader, n_classes)
        test_score = eval_net(netx, nety, netz, test_loader, n_classes)

        train, val = random_split(train_dataset, [n_train, n_val])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=9,
                                  pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=9, pin_memory=True,
                                drop_last=True)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        if n_classes > 1:
            logging.info('Validation cross entropy: {}'.format(val_score))
            logging.info('Test cross_entropy: {}'.format(test_score))
            writer.add_scalar('Loss/eval', val_score, global_step)
            writer.add_scalar('Loss/test', test_score, global_step)
        else:
            logging.info('Validation Dice Coeff: {}'.format(val_score))
            logging.info('Test Dice Coeff: {}'.format(test_score))
            writer.add_scalar('Dice/eval', val_score, global_step)
            writer.add_scalar('Dice/test', test_score, global_step)

        writer.add_images('images', imgs, global_step)
        writer.add_images('masks/true', true_masks, global_step)
        if n_classes == 1:
            writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        else:
            writer.add_images('masks/pred', masks_pred > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(DIR_CHECKPOINTS)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(netz.state_dict(), DIR_CHECKPOINTS + 'AutoEncoder_CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoints {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the AutoEncoder on images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=3e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load net from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help="Percent of the data used as validation (0-100)")

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {device}'.format(device=device))
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    netx = UNet(n_channels=3, n_classes=1)
    nety = UNet(n_channels=3, n_classes=1)
    netz = AutoEncoder()

    try:
        if args.load:
            train_net(netx=netx,
                      nety=nety,
                      netz=netz,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      img_scale=args.scale,
                      val_percent=args.val / 100,
                      load_args=True,
                      **{"pth_file": args.load})
        else:
            train_net(netx=netx,
                      nety=nety,
                      netz=netz,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      img_scale=args.scale,
                      val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(netz.state_dict(), DIR_CHECKPOINTS + 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    print("Training complete!")

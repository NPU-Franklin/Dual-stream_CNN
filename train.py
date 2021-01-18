import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from parallelunet import ParallelUNet
from test import test_net
from utils import MoNuSegTrainingDataset, MoNuSegTestDataset

os.environ['CUDA_VISIBLE_DIVICES'] = "0, 1, 2"

DIR_CHECKPOINTS = './checkpoints/'


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              img_scale=0.5,
              load_args=False,
              *args):
    n_classes = net.n_classes
    n_channels = net.n_channels

    net = nn.DataParallel(net, device_ids=[0, 1])
    if load_args:
        net.load_state_dict(torch.load(args[0]))
        logging.info('Model loaded from {}'.format(args[0]))
    net.cuda()

    train_dataset = MoNuSegTrainingDataset()
    test_dataset = MoNuSegTestDataset()
    n_test = len(test_dataset)
    n_train = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=9, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=9, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment='LR_{}_BS_{}_SCALE_{}'.format(lr, batch_size, img_scale))
    global_step = 0

    logging.info("""Starting training:
        Epochs:             {}
        Batch size:         {}
        Learning rate:      {}
        Training size:      {}
        Test size:          {}
        Checkpoints:        {}
        Device(s):          {}
        Image scaling:      {}
    """.format(epochs, batch_size, lr, n_train, n_test, save_cp, device.type, img_scale))

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                true_edges = batch['edge']
                assert imgs.shape[1] == n_channels, \
                    'Network has been defined with {} input channels, ' \
                    'but loaded images have {} channels. Please check that ' \
                    'the images are loaded correctly.'.format(n_channels, imgs.shape[1])

                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
                true_edges = true_edges.cuda()

                masks_pred, edges_pred = net(imgs)
                loss1 = criterion(masks_pred, true_masks)
                loss2 = criterion(edges_pred, true_edges)
                total_loss = loss1 + loss2
                writer.add_scalar('Loss/train', total_loss.item(), global_step)

                pbar.set_postfix(**{"loss {batch}": total_loss.item()})

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    score1, score2 = test_net(net, test_loader, n_classes)

                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if n_classes > 1:
                        logging.info('Validation cross entropy for masks: {}'.format(score1))
                        logging.info('Validation cross_entropy for edges: {}'.format(score2))
                        writer.add_scalar('Loss/test_on_masks', score1, global_step)
                        writer.add_scalar('Loss/test_on_edges', score2, global_step)
                    else:
                        logging.info('Validation Dice Coeff for masks: {}'.format(score1))
                        logging.info('Validation Dice Coeff for edges: {}'.format(score2))
                        writer.add_scalar('Dice/test_on_masks', score1, global_step)
                        writer.add_scalar('Dice/test_on_edges', score2, global_step)

                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('edges/true', true_edges, global_step)
                    if n_classes == 1:
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                        writer.add_images('edges/pred', torch.sigmoid(edges_pred) > 0.5, global_step)
                    else:
                        writer.add_images('masks/pred', masks_pred > 0.5, global_step)
                        writer.add_images('edges/pred', edges_pred > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(DIR_CHECKPOINTS)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(), DIR_CHECKPOINTS + 'CP_epoch{}.pth'.format(epoch + 1))
            logging.info('Checkpoints {} saved !'.format(epoch + 1))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the Parallel UNet on images, target masks and target edges',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device {device}'.format(device=device))
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = ParallelUNet(n_channels=3, n_classes=1, bilinear=True, bridge_enable=True)
    logging.info('Network:\n'
                 '\t{} input channels\n'
                 '\t{} output channels (classes)\n'
                 '\t{type} upscaling\n'
                 '\t{state} bridge'.format(net.n_channels, net.n_classes,
                                           type="Bilinear" if net.bilinear else "Transposed conv",
                                           state="Enable" if net.bridge_enable else "Disable"))
    try:
        if args.load:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      img_scale=args.scale,
                      load_args=True,
                      *[args.load])
        else:
            train_net(net=net,
                      epochs=args.epochs,
                      batch_size=args.batchsize,
                      lr=args.lr,
                      img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    print("Training complete!")


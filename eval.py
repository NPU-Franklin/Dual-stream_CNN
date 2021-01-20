import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, n_classes):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

            with torch.no_grad():
                mask_pred = net(imgs)

            if n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

import torch
# noinspection PyPep8Naming
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_dual_stream_net(net, val_loader, n_classes):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(val_loader)
    tot1 = 0
    tot2 = 0

    with tqdm(total=n_val, desc="Validation or test round", unit='batch', leave=False) as pbar:
        for batch in val_loader:
            imgs, true_masks, true_edges = batch['image'], batch['mask'], batch['edge']
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()
            true_edges = true_edges.cuda()

            with torch.no_grad():
                mask_pred, edge_pred = net(imgs)

            if n_classes > 1:
                accuracy1 = F.cross_entropy(mask_pred, true_masks).item()
                accuracy2 = F.cross_entropy(edge_pred, true_edges).item()
            else:
                pred1 = torch.sigmoid(mask_pred)
                pred2 = torch.sigmoid(edge_pred)
                pred1 = (pred1 > 0.5).float()
                pred2 = (pred2 > 0.5).float()
                accuracy1 = dice_coeff(pred1, true_masks).item()
                accuracy2 = dice_coeff(pred2, true_edges).item()
            tot1 += accuracy1
            tot2 += accuracy2
            pbar.update()

    net.train()
    return (tot1 / n_val), (tot2 / n_val)

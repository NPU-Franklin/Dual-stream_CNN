import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, val_loader, n_classes):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_val = len(val_loader)
    tot = 0

    with tqdm(total=n_val, desc="Validation or test round", unit='batch', leave=False) as pbar:
        for batch in val_loader:
            imgs, true_edges = batch['image'], batch['edge']
            imgs = imgs.cuda()
            true_edges = true_edges.cuda()

            with torch.no_grad():
                edge_pred = net(imgs)

            if n_classes > 1:
                tot += F.cross_entropy(edge_pred, true_edges).item()
            else:
                pred = torch.sigmoid(edge_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_edges).item()
            pbar.update()

    net.train()
    return tot / n_val

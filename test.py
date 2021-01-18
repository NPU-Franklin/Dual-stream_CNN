import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def test_net(net, loader, n_classes):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    n_test = len(loader)

    with tqdm(total=n_test, desc="Validation round", unit='batch', leave=False) as pbar:
        for batch in loader:
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
                accuracy1 = dice_coeff(pred1, true_masks).item()
                accuracy2 = dice_coeff(pred2, true_edges).item()
            pbar.update()

    net.train()
    return (accuracy1 / n_test), (accuracy2 / n_test)

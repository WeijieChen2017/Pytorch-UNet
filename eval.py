import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            # mask_4 = true_masks
            # mask_3 = avg_pool2d(mask_4, kernel_size=(1, 1), stride=2)
            # mask_2 = avg_pool2d(mask_3, kernel_size=(1, 1), stride=2)
            # mask_1 = avg_pool2d(mask_2, kernel_size=(1, 1), stride=2)

            # mask_4 = mask_4.to(device=device, dtype=mask_type)
            # mask_3 = mask_3.to(device=device, dtype=mask_type)
            # mask_2 = mask_2.to(device=device, dtype=mask_type)
            # mask_1 = mask_1.to(device=device, dtype=mask_type)

            with torch.no_grad():
                # mask_pred = net(imgs)
                [z1, z2, z3, z4] = net(imgs)

                # loss_1 = criterion(z1, mask_1)
                # loss_2 = criterion(z2, mask_2)
                # loss_3 = criterion(z3, mask_3)
                # loss_4 = criterion(z4, mask_4)


            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(z4)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

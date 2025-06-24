import torch
import numpy as np
import time
import matplotlib.pyplot as plt

def pixel_accuracy(preds, target, ignore_index=255):
    mask = target != ignore_index
    correct = (preds == target) & mask
    return correct.sum().float() / mask.sum().float()


def intersection_and_union(output, target, num_classes, ignore_index=255):
    with torch.no_grad():
        _, preds = torch.max(output, 1)
        preds = preds.view(-1)
        target = target.view(-1)
        mask = target != ignore_index

        preds = preds[mask]
        target = target[mask]

        intersection = preds[preds == target]
        area_intersection = torch.histc(intersection.float(), bins=num_classes, min=0, max=num_classes-1)
        area_pred = torch.histc(preds.float(), bins=num_classes, min=0, max=num_classes-1)
        area_target = torch.histc(target.float(), bins=num_classes, min=0, max=num_classes-1)
        area_union = area_pred + area_target - area_intersection
        return area_intersection, area_union


def compute_mIoU(intersection, union):
    iou = intersection / (union + 1e-10)
    return torch.mean(iou)


def dice_score(output, target, num_classes, ignore_index=255):
    with torch.no_grad():
        if output.dim() == 4:
            _, preds = torch.max(output, 1)  # [B, H, W]
        else:
            preds = output  # [B, H, W]

        # Mask out ignore_index
        mask = target != ignore_index

        preds = preds[mask]
        target = target[mask]

        if preds.numel() == 0:
            return torch.tensor(0.0, device=output.device)

        preds = torch.nn.functional.one_hot(preds, num_classes=num_classes).float()
        target = torch.nn.functional.one_hot(target, num_classes=num_classes).float()

        intersection = (preds * target).sum(dim=0)
        union = preds.sum(dim=0) + target.sum(dim=0)

        dice = (2. * intersection + 1e-7) / (union + 1e-7)
        return dice.mean()


# ---------- Helper -----------------------------------------------------------
def decode_segmap(label_tensor, label_colors, ignore_index=255):

    label_np = label_tensor.cpu().numpy()
    h, w = label_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    # mapping ID -> RGB
    id2color = {lab.ID: lab.color for lab in label_colors.list_}

    for id_, color in id2color.items():
        mask = label_np == id_
        color_img[mask] = color

    color_img[label_np == ignore_index] = (0, 0, 0)
    return color_img


# ---------- VISUALIZATION --------------------------------------------------
def show_predictions_triplet(model,
                             dataloader,
                             device,
                             label_colors,
                             num_images=10,
                             denorm=None):
    model.eval()
    shown = 0
    with torch.no_grad():
        for imgs, gt in dataloader:
            imgs = imgs.to(device)
            gt   = gt.to(device)

            with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
                logits = model(imgs)
                if isinstance(logits, tuple):   
                    logits = logits[0]
                preds = torch.argmax(logits, dim=1)   # [B,H,W]

            for i in range(imgs.size(0)):
                # ---------- input ----------
                img_np = imgs[i].cpu()
                if denorm is not None:
                    img_np = denorm(img_np)        
                else:
                    img_np = img_np.permute(1, 2, 0).numpy()
                    if img_np.max() <= 1:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)

                # ---------- MASK ----------
                gt_color   = decode_segmap(gt[i],   label_colors)
                pred_color = decode_segmap(preds[i], label_colors)

                # ---------- plot ----------
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title('Input')
                plt.imshow(img_np); plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.title('Ground Truth')
                plt.imshow(gt_color); plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.title('Prediction')
                plt.imshow(pred_color); plt.axis('off')

                plt.show()

                shown += 1
                if shown >= num_images:
                    return
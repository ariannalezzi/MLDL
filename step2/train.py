#TODO: Define here your training and validation loops.
from tqdm import tqdm
import torch
import torch.nn as nn
from utils_p import *

def train_one_epoch(model, train_loader, optimizer, base_lr, epoch, epochs, device):

    model.train()
    total_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{epochs}")

    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        try:
            outputs, _, _ = model(images)
            if isinstance(outputs, tuple):
              loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, labels.squeeze(1).long())
            else:
              loss = nn.CrossEntropyLoss(ignore_index=255)(outputs, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {i}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        current_iter = epoch * len(train_loader) + i
        poly_lr_scheduler(optimizer, base_lr, current_iter, max_iter=epochs * len(train_loader))

    avg_loss = total_loss / len(train_loader)
    print(f" Avg Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, test_loader, num_classes, device, best_miou):
    import numpy as np
    from tqdm import tqdm

    model.eval()
    hist = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.cpu().numpy()

            try:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Handle tuple output
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("OOM during validation, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            for p, l in zip(preds, labels):
                hist += fast_hist(l.flatten(), p.flatten(), num_classes)

    ious = per_class_iou(hist)
    miou = np.nanmean(ious)
    print(f" Validation mIoU: {miou:.4f}")

    if miou > best_miou:
        print(" New best mIoU found!")
        best_miou = miou

    return best_miou, miou, ious

  

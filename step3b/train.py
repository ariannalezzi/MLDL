#TODO: Define here your training and validation loops.
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils_p import *



def train_one_epoch(model, train_loader, optimizer, base_lr, epoch, epochs, device, checkpoint_dir, start_batch=0):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    max_iter = epochs * len(train_loader)
    
    
    for i, (images, labels) in enumerate(train_loader):
        if i < start_batch:
            continue 

        try:
            print(f"\n Epoch {epoch + 1}/{epochs} - Batch {i + 1}/{len(train_loader)}")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs, _, _ = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            labels = labels.squeeze(1).long()
            loss = criterion(outputs, labels)
            print(f"Loss: {loss.item()}")
       
            if not torch.isfinite(loss):
                print(f"Non-finite loss at Epoch {epoch}, Batch {i}")
                continue

            loss.backward()
            # Debug: check gradient explosion
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 1e3:
                        print(f"Error in Gradient in '{name}' - norm: {grad_norm}")
            optimizer.step()

            current_iter = epoch * len(train_loader) + i
            poly_lr_scheduler(optimizer, base_lr, current_iter, max_iter=epochs * len(train_loader))

            total_loss += loss.item()


        except RuntimeError as e:
            print(f"RuntimeError al batch {i}: {e}")
            torch.cuda.empty_cache()
            checkpoint_path = os.path.join(checkpoint_dir, f"crash_bisenet_ep{epoch}_batch{i}.pt")
            torch.save({
                'epoch': epoch,
                'batch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item() if 'loss' in locals() else None,
                'images': images.cpu(),     
                'labels': labels.cpu()
            }, checkpoint_path)
            print(f"Crash store in: {checkpoint_path}")
            continue

    checkpoint_path = os.path.join(checkpoint_dir, f"bisenet_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss / len(train_loader)
    }, checkpoint_path)
    print(f"Checkpoint epoch stored: {checkpoint_path}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n End epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")
    return avg_loss



def validate(model, test_loader, num_classes, device, best_miou):

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

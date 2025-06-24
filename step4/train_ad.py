from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np #added for validation 
from utils_p import *

def train_one_epoch(seg_model, d_model, 
                    source_loader, target_loader,
                    seg_optimizer, d_optimizer,
                    seg_loss_fn, adv_loss_fn,
                    base_lr, epoch, epochs, device, lambda_adv=0.001):

    seg_model.train()
    d_model.train()

    total_seg_loss = 0.0
    total_adv_loss = 0.0
    total_d_loss = 0.0

    print(f"\nEpoch {epoch + 1}/{epochs}")

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    num_batches = min(len(source_loader), len(target_loader))

    for i in tqdm(range(num_batches)):
        try:
            # === Load Source Batch ===
            source_images, source_labels = next(source_iter)
            source_images = source_images.to(device)
            source_labels = source_labels.to(device)

            # === Load Target Batch ===
            target_images, _ = next(target_iter)
            target_images = target_images.to(device)

            # === Forward Source ===
            seg_optimizer.zero_grad()
            outputs_src, _, _ = seg_model(source_images)

            loss_seg = seg_loss_fn(outputs_src, source_labels.squeeze(1).long())
            loss_seg.backward()
            seg_optimizer.step()

            # === Forward Target ===
            seg_optimizer.zero_grad()
            outputs_tgt, _, _ = seg_model(target_images)
            preds_tgt = torch.softmax(outputs_tgt, dim=1)

            # Freeze discriminator during adversarial step
            for param in d_model.parameters():
                param.requires_grad = False

            d_out_tgt = d_model(preds_tgt)
            adv_labels = torch.ones_like(d_out_tgt).float().to(device)  # Trick: want target to look like source
            loss_adv = adv_loss_fn(d_out_tgt, adv_labels)
            loss_adv = lambda_adv * loss_adv
            loss_adv.backward()
            seg_optimizer.step()

            # === Train Discriminator ===
            d_optimizer.zero_grad()
            for param in d_model.parameters():
                param.requires_grad = True

            # Forward source through D
            preds_src = torch.softmax(outputs_src.detach(), dim=1)
            d_out_src = d_model(preds_src)
            src_domain_labels = torch.ones_like(d_out_src).float().to(device)
            loss_d_src = adv_loss_fn(d_out_src, src_domain_labels)

            # Forward target through D
            preds_tgt = torch.softmax(outputs_tgt.detach(), dim=1)
            d_out_tgt = d_model(preds_tgt)
            tgt_domain_labels = torch.zeros_like(d_out_tgt).float().to(device)
            loss_d_tgt = adv_loss_fn(d_out_tgt, tgt_domain_labels)

            loss_d = 0.5 * (loss_d_src + loss_d_tgt)
            loss_d.backward()
            d_optimizer.step()

            # === Track losses ===
            total_seg_loss += loss_seg.item()
            total_adv_loss += loss_adv.item()
            total_d_loss += loss_d.item()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch {i}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        current_iter = epoch * num_batches + i
        poly_lr_scheduler(seg_optimizer, base_lr, current_iter, max_iter=epochs * num_batches)

    print(f" Avg Seg Loss: {total_seg_loss / num_batches:.4f} | "
          f"Adv Loss: {total_adv_loss / num_batches:.4f} | "
          f"D Loss: {total_d_loss / num_batches:.4f}")



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

#TODO: Define here your training and validation loops.
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils_p import *
from dataset_custom.labels import GTA5Labels_TaskCV2017


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        # CrossEntropyLoss expects class indices, not one-hot
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # print(f"\n[Batch {batch_idx}]")

            # print(f"Original images shape: {images.shape}")
            # print(f"Original targets shape: {targets.shape}")

            images, targets = images.to(device), targets.to(device).long()

            with torch.autocast(device.type if device.type != 'mps' else 'cpu'):
                outputs = model(images)
                # print(f"Model output type: {type(outputs)}")

                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                # print(f"Outputs shape after unpacking (if any): {outputs.shape}")

                loss = criterion(outputs, targets)
                # print(f"Loss computed: {loss.item()}")

                preds = torch.argmax(outputs, dim=1)
                # print(f"Predictions shape: {preds.shape}")

            acc = pixel_accuracy(preds, targets)
            # print(f"Pixel Accuracy: {acc.item()}")

            inter, union = intersection_and_union(outputs, targets, num_classes)
            # print(f"Intersection: {inter}")
            # print(f"Union: {union}")

            iou = compute_mIoU(inter, union)
            # print(f"mIoU: {iou.item()}")

            dice = dice_score(preds, targets, num_classes)
            # print(f"Dice Score: {dice.item()}")

            total_loss += loss.item() * images.size(0)
            total_acc += acc * images.size(0)
            total_miou += iou * images.size(0)
            total_dice += dice * images.size(0)
            total_samples += images.size(0)

    return (
        total_loss / total_samples,
        total_acc / total_samples,
        total_miou / total_samples,
        total_dice / total_samples)

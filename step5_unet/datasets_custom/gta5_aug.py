import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets_custom.labels import GTA5Labels_TaskCV2017

class GTA5(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.image_dir = os.path.join(root, 'images')
        self.label_dir = os.path.join(root, 'labels')

        self.image_paths = sorted([
            os.path.join(self.image_dir, f)
            for f in os.listdir(self.image_dir)
            if f.endswith('.png')
        ])
        
        self.label_paths = sorted([
            os.path.join(self.label_dir, f)
            for f in os.listdir(self.label_dir)
            if f.endswith('.png')
        ])
        
        assert len(self.image_paths) == len(self.label_paths), f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

        # Costruisce dinamicamente il mapping ID -> trainID
        self.id_to_trainid = {label.ID: i for i, label in enumerate(GTA5Labels_TaskCV2017.list_)}
        
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])

        img = np.array(img)           # convert PIL Image to NumPy array
        label = np.array(label)       # convert label to NumPy array

        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask'].long()

        mapped_label = torch.full_like(label, 255)
        for id_, train_id in self.id_to_trainid.items():
            mapped_label[label == id_] = train_id

        return img, mapped_label

    def __len__(self):
        return len(self.image_paths)
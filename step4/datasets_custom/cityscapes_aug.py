import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class CityScapes(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'gtFine', split)

        self.image_paths = []
        self.label_paths = []

        for city in os.listdir(self.image_dir):
            city_img_folder = os.path.join(self.image_dir, city)
            city_label_folder = os.path.join(self.label_dir, city)

            for filename in os.listdir(city_img_folder):
                if filename.endswith('leftImg8bit.png'):
                    img_path = os.path.join(city_img_folder, filename)
                    label_filename = filename.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
                    label_path = os.path.join(city_label_folder, label_filename)

                    if os.path.exists(label_path):
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

        print(f"Loaded {len(self.image_paths)} images for split: {split}")

    def __getitem__(self, idx):
        img = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        label = np.array(Image.open(self.label_paths[idx]))

        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
            if not torch.is_tensor(label):
                label = torch.tensor(label)  # fallback if something went wr
            label = label.long()  # ✅ ensure correct dtype

        return img, label

    def __len__(self):
        return len(self.image_paths)

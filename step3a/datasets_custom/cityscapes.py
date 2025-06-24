from torch.utils.data import Dataset

# TODO: implement here your custom dataset class for Cityscapes


import os
from PIL import Image
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
from torch.utils.data import Dataset

class CityScapes(Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Directories
        self.image_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'gtFine', split)

        # Gather all image-label path pairs
        self.image_paths = []
        self.label_paths = []

        # Traverse city folders
        for city in os.listdir(self.image_dir):
            city_img_folder = os.path.join(self.image_dir, city)
            city_label_folder = os.path.join(self.label_dir, city)

            for filename in os.listdir(city_img_folder):
                if filename.endswith('leftImg8bit.png'):
                    img_path = os.path.join(city_img_folder, filename)
                    label_filename = filename.replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')
                    label_path = os.path.join(city_label_folder, label_filename)

                    if os.path.exists(label_path):  #add only if path exists - si può anche togliere credo
                        self.image_paths.append(img_path)
                        self.label_paths.append(label_path)

        print(f"Loaded {len(self.image_paths)} images for split: {split}") #debug purpose

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = Image.open(self.label_paths[idx])

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return len(self.image_paths)







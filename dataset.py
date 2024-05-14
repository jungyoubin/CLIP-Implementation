import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class ClipDataset(Dataset):
    def __init__(self, img_dir, captions, tokenizer, transform=None, device=torch.device("cpu")):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.device = device

        with open(captions, 'r') as f:
            annotations = json.load(f)['annotations']
            self.annotations = annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.annotations[idx]['image_id']).zfill(12) + '.jpg')
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        caption = self.annotations[idx]['caption']
        return {
            'image': image.to(self.device),
            'caption': caption
        }

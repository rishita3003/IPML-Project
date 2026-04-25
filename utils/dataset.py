import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class DehazeDataset(Dataset):
    def __init__(self, hazy_dir, clean_dir):
        self.hazy_paths = sorted([os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir)])
        self.clean_dir = clean_dir

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        hazy_path = self.hazy_paths[idx]
        filename = os.path.basename(hazy_path)

        # Match clean image
        clean_name = filename.split('_')[0] + ".png"
        clean_path = os.path.join(self.clean_dir, clean_name)

        hazy = Image.open(hazy_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        return self.transform(hazy), self.transform(clean)
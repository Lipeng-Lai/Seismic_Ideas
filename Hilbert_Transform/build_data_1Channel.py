import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandGaussianNoised, RandZoomd,
    RandAffined, RandShiftIntensityd, RandAdjustContrastd, RandGaussianSmoothd,
    ToTensord, Rand2DElasticd
)

class MyDataset(Dataset):
    def __init__(self, image_path1, image_path2, label_path1 ,transform=None):
        super(MyDataset, self).__init__()
        
        self.image_path1 = sorted(glob.glob(os.path.join(image_path1, '*.npy')))
        self.image_path2 = sorted(glob.glob(os.path.join(image_path2, '*.npy')))
        self.label_path1 = sorted(glob.glob(os.path.join(label_path1, '*.npy')))
        
        self.transform = transform
        
    def __len__(self):
        return len(self.image_path1)
    

    def __getitem__(self, index):
        image1 = np.load(self.image_path1[index]).astype(np.float32)
        image2 = np.load(self.image_path2[index]).astype(np.float32)
        image = np.stack([image1, image2], axis=0)
        
        label1 = np.load(self.label_path1[index]).astype(np.float32)
        label = np.expand_dims(label1, axis=0)
        # 拼成dict形式用于MONAI
        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["label"]
    
def get_transforms(is_train=True):
    if is_train:
        return Compose([
            # RandRotate90d(keys=["image", "label"], prob=0.3, spatial_axes=(1, 2)), # 旋转
            # RandFlipd(keys=["image", "label"], prob=0.3, spatial_axis=0), # 翻转
            # RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01), # 高斯噪声
            # RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)),  # 模糊
            # RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),             # 亮度偏移
            # RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.9, 1.1)),        # 对比度增强
            ToTensord(keys=["image", "label"]),
        ])
    else:
        return Compose([
            ToTensord(keys=["image", "label"]),
        ])

if __name__ == "__main__":

    image_path1 = '/home/wwd/deeplearning/data/image'
    image_path2 = '/home/wwd/deeplearning/data/multiple'
    
    label_path1 = '/home/wwd/deeplearning/data/label'
    
    full_dataset = MyDataset(image_path1, image_path2, label_path1, transform=get_transforms(is_train=True))

    val_ratio = 0.2
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = get_transforms(is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))


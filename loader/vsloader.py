import random

import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

from pose_utils import load_pose_table, relative_pose


class LoadVSTrainDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.info_list = load_pose_table(self.root + "/label.csv")
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):
        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)

        img1 = Image.open(
            self.root + "/Image/" + str(info1[0])
        ).convert("RGB")
        img2 = Image.open(
            self.root + "/Image/" + str(info2[0])
        ).convert("RGB")

        pose1 = np.asarray(info1[1:7], dtype=np.float64)
        pose2 = np.asarray(info2[1:7], dtype=np.float64)
        label = relative_pose(pose1, pose2)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomErasing(
                p=1,
                scale=(0.02, 0.03),
                value=(255, 0, 0),
            ),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image1 = image_transform(img1)
        image2 = image_transform(img2)

        return {
            "image1": image1,
            "image2": image2,
            "label": label,
            "name1": info1[0],
            "name2": info2[0],
            "pose1": pose1,
            "pose2": pose2,
        }

    def __len__(self):
        return self.image_num


class LoadVSTestDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.info_list = load_pose_table(self.root + "/label.csv")
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):
        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)

        img1 = Image.open(
            self.root + "/Image/" + str(info1[0])
        ).convert("RGB")
        img2 = Image.open(
            self.root + "/Image/" + str(info2[0])
        ).convert("RGB")

        pose1 = np.asarray(info1[1:7], dtype=np.float64)
        pose2 = np.asarray(info2[1:7], dtype=np.float64)
        label = relative_pose(pose1, pose2)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image1 = image_transform(img1)
        image2 = image_transform(img2)

        return {
            "image1": image1,
            "image2": image2,
            "label": label,
            "name1": info1[0],
            "name2": info2[0],
            "pose1": pose1,
            "pose2": pose2,
        }

    def __len__(self):
        return self.image_num
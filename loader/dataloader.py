import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

from pose_utils import load_pose_table, relative_pose


class LoadTrainDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.im_list = os.listdir(data_root + "/im")
        self.gt_list = os.listdir(data_root + "/gt")
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        seed = random.randint(1, 999)

        img_folder = self.root + "/im/" + self.im_list[item]
        label_folder = self.root + "/gt/" + self.gt_list[item]
        img_path_list = os.listdir(img_folder)

        img_list = []
        label_list = []

        for _ in range(4):
            img_name = random.choice(img_path_list)
            img_list.append(img_folder + "/" + img_name)
            label_list.append(
                label_folder + "/" + img_name.split(".")[0] + ".png"
            )

        images = [
            Image.open(path).resize((112, 112))
            for path in img_list
        ]
        labels = [
            Image.open(path).resize((112, 112))
            for path in label_list
        ]

        new_image = Image.new("RGB", (224, 224))
        new_label = Image.new("L", (224, 224))
        positions = [(0, 0), (112, 0), (0, 112), (112, 112)]

        for image, label, position in zip(images, labels, positions):
            new_image.paste(image, position)
            new_label.paste(label, position)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.5,
                saturation=0.5,
                hue=0.2,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        label_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        torch.manual_seed(seed)
        new_image = image_transform(new_image)

        torch.manual_seed(seed)
        new_label = label_transform(new_label)

        return {
            "image": new_image,
            "label": new_label,
        }

    def __len__(self):
        return self.num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.im_list = os.listdir(data_root + "/im")
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        img_folder = self.root + "/im/" + self.im_list[item]
        img_path_list = os.listdir(img_folder)

        img1 = Image.open(
            img_folder + "/" + img_path_list[0]
        ).resize((112, 112))
        img2 = Image.open(
            img_folder + "/" + img_path_list[1]
        ).resize((112, 112))
        img3 = Image.open(
            img_folder + "/" + img_path_list[0]
        ).resize((112, 112))
        img4 = Image.open(
            img_folder + "/" + img_path_list[1]
        ).resize((112, 112))

        new_image = Image.new("RGB", (224, 224))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (112, 0))
        new_image.paste(img3, (0, 112))
        new_image.paste(img4, (112, 112))

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        return {
            "image": image_transform(new_image),
        }

    def __len__(self):
        return self.num


class LoadVSDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.info_list = load_pose_table(self.root + "/label.csv")
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):
        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)

        pose1 = np.asarray(info1[1:7], dtype=np.float64)
        pose2 = np.asarray(info2[1:7], dtype=np.float64)
        label = relative_pose(pose1, pose2)

        img1 = Image.open(
            self.root + "/Image/" + str(info1[0])
        ).convert("RGB")
        img2 = Image.open(
            self.root + "/Image/" + str(info2[0])
        ).convert("RGB")
        img3 = img1
        img4 = img2

        new_image = Image.new("RGB", (224, 224))
        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (112, 0))
        new_image.paste(img3, (0, 112))
        new_image.paste(img4, (112, 112))

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        new_image = image_transform(new_image)

        image12_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        image1 = image12_transform(img1)
        image2 = image12_transform(img2)

        return {
            "image1": image1,
            "image2": image2,
            "label": label,
            "name1": info1[0],
            "name2": info2[0],
            "pose1": pose1,
            "pose2": pose2,
            "image": new_image,
        }

    def __len__(self):
        return self.image_num
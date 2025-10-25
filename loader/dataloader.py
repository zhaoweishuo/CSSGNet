import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import torch
import os
import random


class LoadTrainDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.im_list = os.listdir(data_root + "/im")
        self.gt_list = os.listdir(data_root + "/gt")
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        seed = random.randint(1, 999)
        img_floder = self.root + "/im/"+self.im_list[item]
        label_floder = self.root + "/gt/"+self.gt_list[item]
        img_path_list = os.listdir(img_floder)
        img_list = []
        label_list = []
        for i in range(4):

            img_name = random.choice(img_path_list)
            img_list.append(img_floder+"/"+img_name)
            label_list.append(label_floder+"/"+img_name.split(".")[0]+".png")

        img1 = Image.open(img_list[0]).resize((112, 112))
        img2 = Image.open(img_list[1]).resize((112, 112))
        img3 = Image.open(img_list[2]).resize((112, 112))
        img4 = Image.open(img_list[3]).resize((112, 112))
        label1 = Image.open(label_list[0]).resize((112, 112))
        label2 = Image.open(label_list[1]).resize((112, 112))
        label3 = Image.open(label_list[2]).resize((112, 112))
        label4 = Image.open(label_list[3]).resize((112, 112))

        new_image = Image.new('RGB', (224, 224))
        new_label = Image.new('L', (224, 224))

        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (112, 0))
        new_image.paste(img3, (0, 112))
        new_image.paste(img4, (112, 112))
        new_label.paste(label1, (0, 0))
        new_label.paste(label2, (112, 0))
        new_label.paste(label3, (0, 112))
        new_label.paste(label4, (112, 112))

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


        return {'image': new_image, 'label': new_label}

    def __len__(self):
        return self.num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.im_list = os.listdir(data_root + "/im")
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        img_floder = self.root + "/im/" + self.im_list[item]
        img_path_list = os.listdir(img_floder)

        img1 = Image.open(img_floder + "/" + img_path_list[0]).resize((112, 112))
        img2 = Image.open(img_floder + "/" + img_path_list[1]).resize((112, 112))
        img3 = Image.open(img_floder + "/" + img_path_list[0]).resize((112, 112))
        img4 = Image.open(img_floder + "/" + img_path_list[1]).resize((112, 112))

        new_image = Image.new('RGB', (224, 224))

        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (112, 0))
        new_image.paste(img3, (0, 112))
        new_image.paste(img4, (112, 112))

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        new_image = image_transform(new_image)


        return {'image': new_image}

    def __len__(self):
        return self.num


class LoadVSDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        csv = pd.read_csv(self.root + "/label.csv")
        self.info_list = csv.to_numpy()  # image column
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):

        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)
        label = info2[1:7] - info1[1:7]
        label = label.astype(np.float32)
        img1 = Image.open(self.root + '/Image/' + info1[0])
        img2 = Image.open(self.root + '/Image/' + info2[0])
        img3 = img1
        img4 = img2

        # -------------cosal
        new_image = Image.new('RGB', (224, 224))

        new_image.paste(img1, (0, 0))
        new_image.paste(img2, (112, 0))
        new_image.paste(img3, (0, 112))
        new_image.paste(img4, (112, 112))

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        new_image = image_transform(new_image)

        # -------------vs
        image12_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image12_transform(img1)
        image2 = image12_transform(img2)

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)

        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0],
                'pose1': pose1, 'pose2': pose2, 'image': new_image}

    def __len__(self):
        return self.image_num


if __name__ == '__main__':

    train_dataset = LoadTrainDataset(data_root="../dataset/DUTS_COCO")
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    for batch_idx, data in enumerate(train_data_loader):
        print(batch_idx)

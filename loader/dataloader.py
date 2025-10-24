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
        self.im_list = os.listdir(data_root + "/im")  # 获取所有类别目录名
        self.gt_list = os.listdir(data_root + "/gt")  # 获取所有类别目录名
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        seed = random.randint(1, 999)
        img_floder = self.root + "/im/"+self.im_list[item]  # 获取当前的类别目录名
        label_floder = self.root + "/gt/"+self.gt_list[item]  # 获取当前的类别目录名
        img_path_list = os.listdir(img_floder)  # 获取协同显著的目录下所有图片名字
        img_list = []
        label_list = []
        for i in range(4):
            # 随机抽取四张图片用于拼接
            img_name = random.choice(img_path_list)
            img_list.append(img_floder+"/"+img_name)  # 数据集im后缀jpg和png都有
            label_list.append(label_floder+"/"+img_name.split(".")[0]+".png")  # 数据集gt图全是png后缀

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
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
            transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5, hue=0.2),  # 亮度，对比度，饱和度，色调随机变换
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 需要看图片调试时注释此行
        ])

        label_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
        ])

        torch.manual_seed(seed)  # 固定随机数使图像与标签增强一致
        new_image = image_transform(new_image)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        torch.manual_seed(seed)
        new_label = label_transform(new_label)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        # # 调试时显示图片 反标准化
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # 将标准差设为1（因为乘以1后什么都不变）
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # trans_to_pil_label = transforms.ToPILImage(mode="L")
        # img_pil = trans_to_pil_image(new_image)
        # lab_pil = trans_to_pil_label(new_label)
        # img_pil.show()
        # lab_pil.show()
        # exit()

        return {'image': new_image, 'label': new_label}

    def __len__(self):
        """返回数据长度"""
        return self.num


class LoadTestDataset(data.Dataset):
    def __init__(self, data_root):
        self.root = data_root
        self.im_list = os.listdir(data_root + "/im")  # 获取所有类别目录名
        self.resize = 224
        self.num = len(self.im_list)

    def __getitem__(self, item):
        img_floder = self.root + "/im/" + self.im_list[item]  # 获取当前的类别目录名
        img_path_list = os.listdir(img_floder)  # 获取协同显著的目录下所有图片名字

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
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 需要看图片调试时注释此行
        ])

        new_image = image_transform(new_image)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        # # 调试时显示图片 反标准化
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # 将标准差设为1（因为乘以1后什么都不变）
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # trans_to_pil_label = transforms.ToPILImage(mode="L")
        # img_pil = trans_to_pil_image(new_image)
        # lab_pil = trans_to_pil_label(new_label)
        # img_pil.show()
        # lab_pil.show()
        # exit()

        return {'image': new_image}

    def __len__(self):
        """返回数据长度"""
        return self.num


class LoadVSDataset(data.Dataset):
    def __init__(self, data_root):
        # 读取文件list
        self.root = data_root
        csv = pd.read_csv(self.root + "/label.csv")
        self.info_list = csv.to_numpy()  # image column
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):
        """建立字典映射"""
        # 1.从文件中读取一个数据（例如，使用numpy.fromfile、PIL.Image.open）。
        # 2.对数据进行预处理（如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)
        label = info2[1:7] - info1[1:7]  # 目标位姿-当前位姿
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
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 需要看图片调试时注释此行
        ])

        new_image = image_transform(new_image)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        # -------------vs
        image12_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image12_transform(img1)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理
        image2 = image12_transform(img2)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)

        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0],
                'pose1': pose1, 'pose2': pose2, 'image': new_image}

    def __len__(self):
        """返回数据长度"""
        return self.image_num


if __name__ == '__main__':
    # 调试训练数据加载
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

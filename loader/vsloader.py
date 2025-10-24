import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import torch
import random


class LoadVSTrainDataset(data.Dataset):
    def __init__(self, data_root):
        # 读取文件list
        self.root = data_root
        csv = pd.read_csv(self.root+"/label.csv")
        self.info_list = csv.to_numpy()  # image column
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):
        """建立字典映射"""
        # 1.从文件中读取一个数据（例如，使用numpy.fromfile、PIL.Image.open）。
        # 2.对数据进行预处理（如torchvision.Transform）。
        # 3.返回数据对（例如图像和标签）。
        # img_name = random.choice(img_path_list)
        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)
        img1 = Image.open(self.root+'/Image/'+info1[0])
        img2 = Image.open(self.root+'/Image/'+info2[0])
        label = info2[1:7]-info1[1:7]  # 目标位姿-当前位姿
        label = label.astype(np.float32)  # 不转换无法返回

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.RandomErasing(p=1, scale=(0.02, 0.03), value=(255, 0, 0)),  # 图像随机遮挡 需要先toTensor
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度，对比度，饱和度，色调随机变换
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image_transform(img1)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理
        image2 = image_transform(img2)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)

        # # 调试时显示图片 反标准化
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # 将标准差设为1（因为乘以1后什么都不变）
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # image1 = trans_to_pil_image(image1)
        # image2 = trans_to_pil_image(image2)
        # image1.show()
        # image2.show()
        # exit()

        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0], 'pose1': pose1, 'pose2': pose2}

    def __len__(self):
        """返回数据长度"""
        return self.image_num


class LoadVSTestDataset(data.Dataset):
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
        img1 = Image.open(self.root + '/Image/' + info1[0])
        img2 = Image.open(self.root + '/Image/' + info2[0])
        label = info2[1:7]-info1[1:7]   # 目标位姿-当前位姿
        label = label.astype(np.float32)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),  # 将图像转为Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image_transform(img1)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理
        image2 = image_transform(img2)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)

        # # 调试时显示图片 反标准化
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # 将标准差设为1（因为乘以1后什么都不变）
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # image1 = trans_to_pil_image(image1)
        # image2 = trans_to_pil_image(image2)
        # image1.show()
        # image2.show()
        # exit()

        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0], 'pose1': pose1, 'pose2': pose2}

    def __len__(self):
        """返回数据长度"""
        return self.image_num


if __name__ == '__main__':
    # 调试训练数据加载
    vs_dataset = LoadVSTrainDataset(data_root="../dataset/10000")
    vs_data_loader = data.DataLoader(
        dataset=vs_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    for batch_idx, data in enumerate(vs_data_loader):
        print(data['pose1'][0])
        print(data['pose2'][0])
        print(data['label'][0])
        # print(data['name1'][0])
        # print(data['name2'][0])
        # # 调试时显示图片 反标准化
        # image1 = data['image1'][0]
        # image2 = data['image2'][0]
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),  # 将标准差设为1（因为乘以1后什么都不变）
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # image1 = trans_to_pil_image(image1)
        # image2 = trans_to_pil_image(image2)
        # image1.show()
        # image2.show()
        exit()


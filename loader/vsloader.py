import pandas as pd
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import torch
import random


class LoadVSTrainDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        csv = pd.read_csv(self.root+"/label.csv")
        self.info_list = csv.to_numpy()  # image column
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):

        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)
        img1 = Image.open(self.root+'/Image/'+info1[0])
        img2 = Image.open(self.root+'/Image/'+info2[0])
        label = info2[1:7]-info1[1:7]
        label = label.astype(np.float32)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=1, scale=(0.02, 0.03), value=(255, 0, 0)),
            transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image_transform(img1)
        image2 = image_transform(img2)

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)


        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0], 'pose1': pose1, 'pose2': pose2}

    def __len__(self):

        return self.image_num


class LoadVSTestDataset(data.Dataset):
    def __init__(self, data_root):

        self.root = data_root
        csv = pd.read_csv(self.root + "/label.csv")
        self.info_list = csv.to_numpy()  # image column
        self.resize = 224
        self.image_num = len(self.info_list)

    def __getitem__(self, item):

        info1 = self.info_list[item]
        info2 = random.choice(self.info_list)
        img1 = Image.open(self.root + '/Image/' + info1[0])
        img2 = Image.open(self.root + '/Image/' + info2[0])
        label = info2[1:7]-info1[1:7]
        label = label.astype(np.float32)

        image_transform = transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1 = image_transform(img1)
        image2 = image_transform(img2)

        pose1 = info1[1:7].astype(np.float32)
        pose2 = info2[1:7].astype(np.float32)



        return {'image1': image1, 'image2': image2, 'label': label, 'name1': info1[0], 'name2': info2[0], 'pose1': pose1, 'pose2': pose2}

    def __len__(self):

        return self.image_num


if __name__ == '__main__':

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

        # image1 = data['image1'][0]
        # image2 = data['image2'][0]
        # trans_to_pil_image = transforms.Compose([
        #     transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        #     transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
        #     transforms.ToPILImage(mode="RGB")
        # ])
        # image1 = trans_to_pil_image(image1)
        # image2 = trans_to_pil_image(image2)
        # image1.show()
        # image2.show()
        exit()


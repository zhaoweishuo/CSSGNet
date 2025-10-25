import argparse
import torch
import loader
from torch.utils import data
import model
from tqdm import tqdm
import numpy as np
import loss
import torchvision.transforms as transforms
from PIL import Image

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(model, device, data_loader):
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            image1 = data['image1'].to(device)
            image2 = data['image2'].to(device)
            label = data['label'].numpy()
            pose1 = data['pose1'].numpy()
            pose2 = data['pose2'].numpy()


            y_hat = model(image1, image2)
            y_hat = y_hat.cpu().numpy()
            print(pose1)
            print(pose2)
            print(label)
            print(y_hat)


            trans_to_pil_image = transforms.Compose([
                transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
                transforms.ToPILImage(mode="RGB")
            ])
            image1 = trans_to_pil_image(image1[0])
            image2 = trans_to_pil_image(image2[0])
            image1.show()
            image2.show()

            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--device', default=0, type=int, help="gpu serial number")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_folder', default="./dataset/10000", type=str)
    parser.add_argument('--pth_path', default="./checkpoint/vs32.pth", type=str)

    args = parser.parse_args()

    DEVICE = try_gpu(args.device)
    BATCH_SIZE = args.batch_size
    PTH_PATH = args.pth_path
    TEST_FOLDER = args.test_folder

    MODEL = model.Vs(pretrained=False).to(DEVICE)
    MODEL.load_state_dict(torch.load(PTH_PATH))  # 加载训练好的模型

    dataset = loader.LoadVSTestDataset(data_root=TEST_FOLDER)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    train(MODEL, DEVICE, data_loader)

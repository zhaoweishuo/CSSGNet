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


def train(model, device, data_loader, loss_func, optimizer, scheduler, batch_size, epoch):
    model.train()
    for epoch_num in range(1, epoch + 1):
        total_num = len(data_loader.dataset)
        train_bar = tqdm(total=total_num)
        epoch_loss = np.array([])
        for batch_idx, data in enumerate(data_loader):
            label = data['label'].to(device)
            image1 = data['image1'].to(device)
            image2 = data['image2'].to(device)

            y_hat = model(image1, image2)
            loss = loss_func(y_hat, label)


            epoch_loss = np.append(epoch_loss, loss.item())

            progress = batch_size
            train_bar.set_description(f'Epoch [{epoch_num}/{epoch}]')
            train_bar.set_postfix(loss=epoch_loss.mean(), lr=optimizer.param_groups[0]['lr'])
            train_bar.update(progress)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with open("./log/vs_train_log.txt", 'a') as file_object:
            file_object.write(
                "epoch:{},lr:{:.8f},loss:{:.5f}\n".format(
                    epoch_num,
                    optimizer.param_groups[0]['lr'],
                    epoch_loss.mean(),
                ))

        scheduler.step()
        train_bar.close()
        torch.save(model.state_dict(), "./checkpoint/vs.pth")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--device', default=0, type=int, help="gpu serial number")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--train_folder', default="./dataset/10000", type=str)
    parser.add_argument('--learn_rate', default=1e-4, type=float)

    args = parser.parse_args()

    DEVICE = try_gpu(args.device)
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    LR = args.learn_rate
    TRAIN_FOLDER = args.train_folder

    LOSS_FUNC = torch.nn.L1Loss()
    MODEL = model.Vs().to(DEVICE)
    OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=LR)
    SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=EPOCH + 10, eta_min=1e-7)

    dataset = loader.LoadVSTrainDataset(data_root=TRAIN_FOLDER)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    train(MODEL, DEVICE, data_loader, LOSS_FUNC, OPTIMIZER, SCHEDULER, BATCH_SIZE, EPOCH)




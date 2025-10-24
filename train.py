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
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def train(model, device, train_data_loader, loss_func, optimizer, scheduler, batch_size, epoch):
    model.train()
    for epoch_num in range(1, epoch + 1):
        total_num = len(train_data_loader.dataset)  # 训练数据总数
        train_bar = tqdm(total=total_num)  # 设置进度条
        epoch_loss = np.array([])  # 当前轮所有批次的loss
        for batch_idx, data in enumerate(train_data_loader):
            label = data['label'].to(device)
            image = data['image'].to(device)

            y_hat = model(image)
            loss = loss_func(y_hat, label)

            # 记录每个batch的损失
            epoch_loss = np.append(epoch_loss, loss.item())

            # 更新进度条数据
            progress = batch_size  # 每次更新一个batch_size的进度
            train_bar.set_description(f'Epoch [{epoch_num}/{epoch}]')
            train_bar.set_postfix(loss=epoch_loss.mean(), lr=optimizer.param_groups[0]['lr'])  # 显示损失与当前学习率
            train_bar.update(progress)

            # 模型参数变更 在每个batch里
            optimizer.zero_grad()  # 梯度归零要在loss.backward()之前
            loss.backward()  # 损失反向传播
            optimizer.step()  # 更新权重要在loss.backward()之后

        # 记录每轮的损失与学习率 每个epoch
        with open("./log/cosal_train_log.txt", 'a') as file_object:
            file_object.write(
                "epoch:{},lr:{:.8f},loss:{:.5f}\n".format(
                    epoch_num,
                    optimizer.param_groups[0]['lr'],
                    epoch_loss.mean(),
                ))

        # 学习率调整 保存参数 每个epoch
        scheduler.step()  # 执行学习率调整策略 要放在epoch循环
        train_bar.close()  # 关闭进度条实例，不然下一个epoch进度条会出现异常
        torch.save(model.state_dict(), "./checkpoint/cosal11.pth")  # 每一轮存一次


if __name__ == '__main__':
    # 参数接收
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--device', default=0, type=int, help="gpu serial number")
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=100, type=int)  # 训练1000  微调100
    parser.add_argument('--train_folder', default="./dataset/sim", type=str)
    parser.add_argument('--learn_rate', default=3e-4, type=float)
    parser.add_argument('--mode', default="refinement", type=str)

    args = parser.parse_args()

    # 参数设置
    DEVICE = try_gpu(args.device)
    BATCH_SIZE = args.batch_size
    EPOCH = args.epoch
    LR = args.learn_rate
    TRAIN_FOLDER = args.train_folder

    # 模型准备
    if args.mode == 'normal':
        MODEL = model.Sal().to(DEVICE)
        OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=LR)
        SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=EPOCH + 10, eta_min=1e-7)
        print("Normal train mode with pretrained model")
    elif args.mode == 'refinement':
        MODEL = model.Sal(pretrained=False).to(DEVICE)
        MODEL.load_state_dict(torch.load("./checkpoint/cosal11.pth", map_location=DEVICE))  # 加载训练好的模型
        OPTIMIZER = torch.optim.Adam(params=MODEL.parameters(), lr=3e-4)
        SCHEDULER = torch.optim.lr_scheduler.StepLR(OPTIMIZER, step_size=10, gamma=0.7)  # step_sizege间隔乘以gamma调整学习率
        print("Refinement train mode with self-trained model")

    LOSS_FUNC = loss.MyLoss()

    # 加载数据
    train_dataset = loader.LoadTrainDataset(data_root=TRAIN_FOLDER)
    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 模型训练
    train(MODEL, DEVICE, train_data_loader, LOSS_FUNC, OPTIMIZER, SCHEDULER, BATCH_SIZE, EPOCH)




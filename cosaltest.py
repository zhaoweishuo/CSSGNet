import argparse
import torch
import loader
from torch.utils import data
import model
import torchvision.transforms as transforms
from PIL import Image


def try_gpu(i=0):

    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def test(model, device, test_data_loader, vsmodel):
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_data_loader):
            x = data['image'].to(device)
            y_hat = model(x)

            pred = y_hat.squeeze(0).cpu()
            image = x.squeeze(0).cpu()


            trans_to_pil_image = transforms.ToPILImage(mode="RGB")
            trans_to_pil_pred = transforms.ToPILImage(mode="L")
            image = trans_to_pil_image(image)
            pred = trans_to_pil_pred(pred)

            # crop cosal img
            cosal1 = pred.crop((0, 0, 112, 112))
            cosal2 = pred.crop((113, 113, 224, 224))
            cosal1 = cosal1.resize((224, 224))
            cosal2 = cosal2.resize((224, 224))
            cosal1 = cosal1.convert('RGB')
            cosal2 = cosal2.convert('RGB')


            # image.show()
            # pred.show()
            # cosal1.show()
            # cosal2.show()

            # trans 2 Tensor
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # 将图像转为Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 需要看图片调试时注释此行
            ])
            cosal1 = image_transform(cosal1)  # 转为tensor并且标准化  转为tensor之后shape变为(channel,w,h) 以便后续进行处理
            cosal2 = image_transform(cosal2)
            cosal1 = cosal1.reshape(1, 3, 224, 224)
            cosal2 = cosal2.reshape(1, 3, 224, 224)
            cosal1 = cosal1.to(device)
            cosal2 = cosal2.to(device)

            # vs
            label = data['label'].numpy()
            pose1 = data['pose1'].numpy()
            pose2 = data['pose2'].numpy()

            y_hat = vsmodel(cosal1, cosal2)
            y_hat = y_hat.cpu().numpy()
            print(pose1)
            print(pose2)
            print(label)
            print(y_hat)

            exit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--device', default=0, type=int, help="gpu serial number")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_folder', default="./dataset/10000", type=str)
    parser.add_argument('--pth_path', default="./checkpoint/cosal11.pth", type=str)
    parser.add_argument('--vspth_path', default="./checkpoint/vs32.pth", type=str)
    args = parser.parse_args()


    DEVICE = try_gpu(args.device)
    BATCH_SIZE = args.batch_size
    TEST_FOLDER = args.test_folder
    PTH_PATH = args.pth_path
    VSPTH_PATH = args.vspth_path


    MODEL = model.Sal(pretrained=False).to(DEVICE)
    MODEL.load_state_dict(torch.load(PTH_PATH))
    VSMODEL = model.Vs(pretrained=False).to(DEVICE)
    VSMODEL.load_state_dict(torch.load(VSPTH_PATH))


    test_dataset = loader.LoadVSDataset(data_root=TEST_FOLDER)
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )


    test(MODEL, DEVICE, test_data_loader, VSMODEL)



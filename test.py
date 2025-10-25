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


def test(model, device, test_data_loader):
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

            # image.show()
            # pred.show()

            pred.save("./dataset/sim/"+str(batch_idx+1)+'.png')




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--device', default=0, type=int, help="gpu serial number")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_folder', default="./dataset/real", type=str)
    parser.add_argument('--pth_path', default="./checkpoint/cosal11.pth", type=str)
    args = parser.parse_args()


    DEVICE = try_gpu(args.device)
    BATCH_SIZE = args.batch_size
    TEST_FOLDER = args.test_folder
    PTH_PATH = args.pth_path


    MODEL = model.Sal(pretrained=False).to(DEVICE)
    MODEL.load_state_dict(torch.load(PTH_PATH))


    test_dataset = loader.LoadTestDataset(data_root=TEST_FOLDER)
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )


    test(MODEL, DEVICE, test_data_loader)


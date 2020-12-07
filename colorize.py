import argparse
import torch
from torchvision.utils import save_image
from network import ColorizeNet
from utils import load_gray, to_rgb

parser = argparse.ArgumentParser(description='colorizes an image')
parser.add_argument('-i', '--img_path', type=str, metavar='', required=True,
                    help='path and/or name of grayscale image to colorize')
parser.add_argument('-r', '--res', type=int, metavar='',
                    help='resizes the input to given resolution {default:360}')
parser.add_argument('-o', '--out_path', type=str, metavar='', required=True,
                    help='name to which the colorized image to be saved')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = ColorizeNet()
model.load_state_dict(
    torch.load('models/model.pth', map_location='cpu')
)


def main():
    args = parser.parse_args()
    img_l = load_gray(args.img_path, shape=args.res)

    model.eval()
    with torch.no_grad():
        img_ab = model(img_l)

    img_rgb = to_rgb(img_l, img_ab)
    save_image(torch.from_numpy(img_rgb.transpose(2, 0, 1)),
               args.out_path)
    print(f'>>> colorized image saved to "{args.out_path}"')


if __name__ == '__main__':
    main()

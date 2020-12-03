import argparse
import torch
from torchvision.utils import save_image
from network import ColorizeNet
from utils import load_gray, to_rgb

parser = argparse.ArgumentParser(description='Colorize a grayscale image')
parser.add_argument('-i', '--img_path', type=str, metavar='', required=True,
                    help='path and/or name of grayscale image to colorize')
parser.add_argument('-s', '--shape', type=int, metavar='',
                    help='saves colorized image to given shape (in pixels)')
parser.add_argument('-o', '--out_name', type=str, metavar='', required=True,
                    help='name to which the colorized image to be saved')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('models/model_0.001.pth', map_location=device)

model = ColorizeNet()
model.load_state_dict(checkpoint['model'])


def main():
    path = 'images/outputs/'
    args = parser.parse_args()
    img_l = load_gray(args.img_path, shape=args.shape)

    model.eval()
    with torch.no_grad():
        img_ab = model(img_l)

    img_rgb = to_rgb(img_l, img_ab)
    save_image(torch.from_numpy(img_rgb.transpose(2, 0, 1)),
               path+args.out_name)
    print(f'\tColorized image saved to "{path}{args.out_name}"')


if __name__ == '__main__':
    main()

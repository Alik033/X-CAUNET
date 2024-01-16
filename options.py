import argparse
import torch
import os

parser = argparse.ArgumentParser()

parser.add_argument('--data_path', default='./UIEB/Train/')

parser.add_argument('--checkpoints_dir', default='./checkpoint/')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--num_images', type=int, default=0)

parser.add_argument('--learning_rate_g', type=float, default=2e-04)

parser.add_argument('--end_epoch', type=int, default=100)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--image_size', type=int ,default=256)

parser.add_argument('--beta1', type=float ,default=0.5)
parser.add_argument('--beta2', type=float ,default=0.999)
parser.add_argument('--wd_g', type=float ,default=0.00005)
parser.add_argument('--wd_d', type=float ,default=0.00000)

parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)

parser.add_argument('--batch_vgg_loss', type=float, default=0.0)
parser.add_argument('--total_vgg_loss', type=float, default=0.0)

parser.add_argument('--batch_log_loss', type=float, default=0.0)
parser.add_argument('--total_log_loss', type=float, default=0.0)

parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)

parser.add_argument('--lambda_mse', type=float, default=1.0)
parser.add_argument('--lambda_vgg', type=float, default=0.02)
parser.add_argument('--lambda_ssim', type=float, default=0.5)

parser.add_argument('--testing_start', type=int, default=1)
parser.add_argument('--testing_end', type=int, default=1)
parser.add_argument('--testing_mode', default="Nat")

parser.add_argument('--testing_dir_inp', default="./UIEB/Test/inp/")
parser.add_argument('--testing_dir_gt', default="./UIEB/Test/gt/")

opt = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)

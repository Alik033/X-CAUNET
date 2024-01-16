import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
import lpips
import torchvision.transforms as transforms
import torch
## compares avg ssim and psnr 
def LPIPS(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """

    with torch.no_grad():

        loss_fn = lpips.LPIPS(net='vgg')
        gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
        gen_paths = sorted(glob(join(gen_dir, "*.*")))
        lpips_arr = []

        # Define the transformation to convert to a tensor
        transform = transforms.ToTensor()


        for gtr_path, gen_path in zip(gtr_paths, gen_paths):
            gtr_f = basename(gtr_path).split('.')[0]
            gen_f = basename(gen_path).split('.')[0]
            if (gtr_f==gen_f):
                # assumes same filenames
                r_im = Image.open(gtr_path).resize(im_res)
                g_im = Image.open(gen_path).resize(im_res)
                # get lpips on RGB channels
                val = loss_fn(transform(r_im),transform(g_im))
                #val = loss_fn(r_im,g_im)
                lpips_arr.append(val.item())
        return np.array(lpips_arr)


if __name__ == '__main__':
    gtr_dir = "./TEST_CLEAN_UIEB/"
    gen_dir = "./UIEB/uieb_43_EMNet/"
    lpips_arr = LPIPS(gtr_dir, gen_dir)
    print(lpips_arr)
    print("LPIPS on {0} samples".format(len(lpips_arr))+"\n")

    print("Mean: {0} std: {1}".format(np.mean(lpips_arr), np.std(lpips_arr))+"\n")

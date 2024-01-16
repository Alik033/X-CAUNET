import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
import torchvision.transforms as transforms

## compares avg ssim and psnr 
def MS_SSIM(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images 
    """

    with torch.no_grad():

        gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
        gen_paths = sorted(glob(join(gen_dir, "*.*")))
        r_arr = []
        MsSSIM = []

        # Define the transformation to convert to a tensor
        transform = transforms.ToTensor()


        for gtr_path, gen_path in zip(gtr_paths, gen_paths):
            gtr_f = basename(gtr_path).split('.')[0]
            gen_f = basename(gen_path).split('.')[0]
            if (gtr_f==gen_f):
                # assumes same filenames
                r_im = Image.open(gtr_path).resize(im_res)
                g_im = Image.open(gen_path).resize(im_res)
                #image to np array
                r_im = np.array(r_im).astype(np.float32)
                g_im = np.array(g_im).astype(np.float32)
                # 1, C, H, W
                r_im = torch.from_numpy(r_im).unsqueeze(0).permute(0, 3, 1, 2) 
                g_im = torch.from_numpy(g_im).unsqueeze(0).permute(0, 3, 1, 2)
                val = ms_ssim(r_im, g_im, data_range=255, size_average=True)        
                MsSSIM.append(val)
        return np.array(MsSSIM)


if __name__ == '__main__':
    gtr_dir = "./TEST_CLEAN_UIEB/"
    gen_dir = "./UIEB/uieb_43_EMNet/"
    arr = MS_SSIM(gtr_dir, gen_dir)
    print("MS-SSIM on {0} samples".format(len(arr))+"\n")
    print("Mean: {0} std: {1}".format(np.mean(arr), np.std(arr))+"\n")
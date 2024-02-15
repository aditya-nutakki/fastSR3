import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
import os, numpy, json
# from unet_time import UNet
from unet_attn import UNet
import torchshow
from time import time

from helpers import *
from config import *
from diffusion import DiffusionModel



def eval(num_samples = batch_size):
    ddpm = DiffusionModel(time_steps = time_steps)
    # ddpm.load_state_dict(torch.load(os.path.join(model_save_dir, "sr_ep_7_64x128.pt")))
    epochs = [18, 54, 87]
    # for ep in epochs:

    ddpm.load_state_dict(torch.load(os.path.join(model_save_dir, f"celeba_sr_ep_143_16x128_t2000.pt")))
    ddpm = ddpm.to(device)

    #     c, h, w = image_dims
    #     assert h == w, f"height and width must be same, got {h} as height and {w} as width"
    #     print(f"Loaded model, trying to sample !")
        
        # loader = get_dataloader(dataset_type="sr", img_sz = h, batch_size = num_samples)
    
    loader = get_dataloader(path = "/mnt/d/work/datasets/celebA_test", hr_sz = hr_sz, lr_sz = lr_sz, batch_size = num_samples)
    # ts = [10, 50, 100, 200, 500, 1000, 1500, 2000]
    ts = [10, 50, 100]
    for i, (hr_img, lr_img) in enumerate(loader):
    
        # x = ddpm.sample(lr_img, num_samples=num_samples)
        # torchshow.save(x, os.path.join(img_save_dir, f"sample_{ep}.jpeg"))

        print(f"done with normal sampling ...")
        print()
        for _t in ts:
            x = ddpm.ddim_sample(lr_img, sample_steps = _t, eta = 1.0)
            # torchshow.save(x, os.path.join(img_save_dir, f"ddim_sample_{_t}.jpeg"))

        torchshow.save(hr_img, os.path.join(img_save_dir, f"hr_img.jpeg"))
        torchshow.save(lr_img, os.path.join(img_save_dir, f"lr_img.jpeg"))

        break



if __name__ == "__main__":
    eval(num_samples = 24)


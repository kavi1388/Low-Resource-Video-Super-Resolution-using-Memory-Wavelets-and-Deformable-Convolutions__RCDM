import torch
import os
from model.rcdm import RCDM
from data.dataset import *
from torch.utils.data import DataLoader
from utils.ssim import *
from utils.vgg_loss import VGGPerceptualLoss
from utils.charb_loss import *
from utils.patchify import *
from config import *
import glob
import torch.optim as optim
from torch.optim import Adam
import piq
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model and dataloader
model = RCDM(num_channels=3,
kernel_size=(3, 3),
padding=(1,1),
activation="relu",
scale=4,
group_of_frames = GOF,num_of_patches=PATCHES).to(device)

epochs = EPOCHS
res_path = RES_PATH
params = PARAMS
PATH = f'{res_path}/{params}/model_best.pth'
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
    print('loaded model')

lr_path = sorted(glob.glob('../REDS/train/train_BI/*/*//*.png'))

gof = GOF
test_data = CustomDatasetInf(lr_path,gof)
# partial = list(range(0, len(train_data), 1))
# trainset_1 = torch.utils.data.Subset(train_data, partial)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=True)

torch.cuda.empty_cache()
gc.collect()


print(f'len of test_loader ={len(test_loader)}')
for epoch in range(epochs):
    
    """### Inference"""
    count = 0
    psnr_val = []
    ssim_val = []
    lpips_val = []
    frame_num = []
    patch_num = []
    num = 0
    model.eval()
    st = time.time()
    with torch.no_grad():
        for batch_num, data in enumerate(test_loader, 0):
            input = data.to(device)
            frame_num.append(batch_num)
            print(f'{input.shape}')
            patch = Patchify(patch_size=(int(test_loader.dataset[0].shape[-2]//math.sqrt(PATCHES)),int(test_loader.dataset[0].shape[-1]//math.sqrt(PATCHES))))
            patch_lr = []
            op_ =[]
            for fr in range(input.shape[1]):
                image = input[:,fr,:,:,:]
                p = patch(image)
                patch_lr.append(p)
            ip_ = torch.stack(patch_lr,dim=2)

            ip_ = ip_.flatten(0,1)
            
            # Modifying target
            if count ==1:
                state = None
            output = model(ip_.cuda())
            state = output[1].detach()
            output = output[0]
            
            output_image = Unpatchify(patch_size=(int(test_loader.dataset[0][0].shape[-2]*4//math.sqrt(PATCHES)),int(test_loader.dataset[0][0].shape[-1]*4//math.sqrt(PATCHES))), image_size=(int(test_loader.dataset[0][0].shape[-2]*4*SCALE//math.sqrt(PATCHES)),int(test_loader.dataset[0][0].shape[-1]*4*SCALE//math.sqrt(PATCHES))))(output.view(1, output.shape[0], output.shape[1], output.shape[2], output.shape[3]))
                


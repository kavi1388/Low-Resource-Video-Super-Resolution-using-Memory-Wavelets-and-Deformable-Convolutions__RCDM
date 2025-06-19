import torch
import os
from model.rcdm import RCDM
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
from utils.ssim import *
from utils.vgg_loss import VGGPerceptualLoss
from utils.charb_loss import *
from utils.patchify import Patchify
from config import *
import glob
import torch.optim as optim
from torch.optim import Adam
import piq
import time
import numpy as np
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate model and dataloader
model = RCDM(num_channels=3,
kernel_size=(3, 3),
padding=(1,1),
activation="relu",
scale=SCALE,
group_of_frames = GOF,num_of_patches=PATCHES).to(device)

epochs = EPOCHS
res_path = RES_PATH
params = PARAMS
PATH = f'{res_path}/{params}/model_best.pth'
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH))
    print('loaded model')

hr_path = sorted(glob.glob('../REDS/train/train_GT/*/*//*.png'))
lr_path = sorted(glob.glob('../REDS/train/train_BI/*/*//*.png'))
# gof = 5
gof = GOF
train_data = CustomDataset(hr_path,lr_path,gof)
# partial = list(range(0, len(train_data), 1))
# trainset_1 = torch.utils.data.Subset(train_data, partial)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

hr_path = sorted(glob.glob('../REDS/val/val_GT/*//*.png'))
lr_path = sorted(glob.glob('../REDS/val/val_BI/*//*.png'))

val_data = CustomDataset(hr_path,lr_path,gof)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False,drop_last=True)
import gc
torch.cuda.empty_cache()
gc.collect()


import warnings
warnings.filterwarnings("ignore")


criterion2 = CharbonnierLoss().cuda()
criterion1 = VGGPerceptualLoss().cuda()

scaler = torch.cuda.amp.GradScaler()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

# Training loop...
"""### Training"""
print(f'len of train_loader ={len(train_loader)}')
print(f'len of val_loader ={len(val_loader)}')
for epoch in range(epochs):
    
    psnr = []
    ssim = []
    lpips = []
    train_loss = 0
    count = 0
    num = 0
    model.train()
    st = time.time()
    
    for batch_num, data in enumerate(train_loader, 0):
        input, target = data[0].to(device), data[1]
        patch = Patchify(patch_size=(int(train_loader.dataset[0][0].shape[-2]//math.sqrt(PATCHES)),int(train_loader.dataset[0][0].shape[-1]//math.sqrt(PATCHES))))
        patch_lr = []
        op_ =[]
        for fr in range(input.shape[1]):
            image = input[:,fr,:,:,:]
            
            p = patch(image)
            patch_lr.append(p)
        ip_ = torch.stack(patch_lr,dim=2)
        
        ip_ = ip_.flatten(0,1)

        patch = Patchify(patch_size=(int(train_loader.dataset[0][1].shape[-2]//math.sqrt(PATCHES)),int(train_loader.dataset[0][1].shape[-1]//math.sqrt(PATCHES))))
        image = target
    
        op_ = patch(image)
        
        op_ = op_.flatten(0,1)
#      
        count+=1
        
        # Modifying target
        if count ==1:
            state = None
        output = model(ip_.cuda())
        state = output[1].detach()
        output = output[0]
#         print(f'op {output.shape}, target {op_.shape} ip {ip_.shape}')
        loss = 1000*criterion2(output, op_.cuda())
        loss.backward(retain_graph=True)
        target = op_
        input = ip_
        
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        psnr.append(calculate_psnr(255.0*output.cpu().permute(2, 3, 1, 0).detach().numpy(), 255.0*target.permute(2, 3, 1, 0).numpy()))
        ssim.append(ssim_calc(output, target.cuda()).detach().cpu())
        
    train_loss /= len(train_loader.dataset)
    psnr_avg= sum(psnr)/len(psnr)
    ssim_avg= sum(ssim)/len(ssim)
    # lpips_avg= sum(lpips)/len(train_loader.dataset)
    psnr_max = max(psnr)
    ssim_max = max(ssim)
    
    
    if ssim_avg > ssim_best:
        ssim_best = ssim_avg
        PATH = f'{res_path}/{params}/train/model_best.pth'
        torch.save(model.state_dict(), PATH)
    
    if not os.path.exists(f'{res_path}/{params}/train/metrics'):
                    os.makedirs(f'{res_path}/{params}/train/metrics')
    with open(f'{res_path}/{params}/train/metrics/{epoch+1}_quality metrics', 'w') as fp:

        fp.write("\n SSIM")
        for item in ssim:
            # write each item on a new line
            fp.write("%s\n" % item)
            num += num
            
        fp.write("\n PSNR")
        for item in psnr:
            # write each item on a new line
            fp.write("%s\n" % item)
            num += num
    
    print("Epoch:{} Training Loss:{:.3f} in {:.2f}\n".format(
        epoch+1, train_loss, time.time()-st))
    print(f'Train PSNR avg {psnr_avg}, PSNR max {psnr_max}, Train SSIM avg {ssim_avg} , SSIM max {ssim_max}')
    """### Validation"""
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
        for batch_num, data in enumerate(val_loader, 0):
            input, target = data[0].to(device), data[1]
            frame_num.append(batch_num)
            patch = Patchify(patch_size=(int(train_loader.dataset[0][0].shape[-2]//math.sqrt(PATCHES)),int(train_loader.dataset[0][0].shape[-1]//math.sqrt(PATCHES))))
            patch_lr = []
            op_ =[]
            for fr in range(input.shape[1]):
                image = input[:,fr,:,:,:]
                p = patch(image)
                patch_lr.append(p)
            ip_ = torch.stack(patch_lr,dim=2)

            ip_ = ip_.flatten(0,1)

            patch = Patchify(patch_size=(int(train_loader.dataset[0][1].shape[-2]//math.sqrt(PATCHES)),int(train_loader.dataset[0][1].shape[-1]//math.sqrt(PATCHES))))
            image = target
            op_ = patch(image)

            op_ = op_.flatten(0,1)

            count+=1
            patch_num.append(range(batch_num*op_.shape[0], batch_num*op_.shape[0]+op_.shape[0]))

            # Modifying target
            if count ==1:
                state = None
            output = model(ip_.cuda())
            state = output[1].detach()
            output = output[0]
            
            target = op_
            input = ip_
    
            psnr_val.append(calculate_psnr(255.0*output.cpu().permute(2, 3, 1, 0).detach().numpy(), 255.0*target.permute(2, 3, 1, 0).numpy()))
            ssim_val.append(ssim_calc(output, target.cuda()).detach().cpu())

        psnr_val_avg= sum(psnr_val)/len(psnr_val)
        ssim_val_avg= sum(ssim_val)/len(ssim_val)
        
        psnr_val_max = max(psnr_val)
        ssim_val_max = max(ssim_val)


        if ssim_val_avg > ssim_val_best:
            ssim_val_best = ssim_val_avg
            PATH = f'{res_path}/{params}/val/model_best.pth'
            torch.save(model.state_dict(), PATH)

        if not os.path.exists(f'{res_path}/{params}/val/metrics'):
                        os.makedirs(f'{res_path}/{params}/val/metrics')
        with open(f'{res_path}/{params}/val/metrics/{epoch+1}_quality metrics', 'w') as fp:

            fp.write("\n SSIM")
            for item in ssim:
                # write each item on a new line
                fp.write("%s\n" % item)
                num += num

            fp.write("\n PSNR")
            for item in psnr:
                # write each item on a new line
                fp.write("%s\n" % item)
                num += num


        print("Epoch:{} Val in {:.2f}\n".format(
            epoch+1, time.time()-st))
        print(f'Val PSNR avg {psnr_val_avg}, PSNR max {psnr_val_max}, Val SSIM avg {ssim_val_avg} , SSIM max {ssim_val_max}')


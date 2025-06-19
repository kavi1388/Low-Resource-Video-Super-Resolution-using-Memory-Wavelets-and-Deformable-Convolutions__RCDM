import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import glob
import piq

class CustomDataset(Dataset):
    def __init__(self, hr_path, lr_path, gof, threshold=0.3):
        self.gof = gof
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.threshold = threshold

    def __len__(self):
        return len(self.hr_path) - self.gof

    def __getitem__(self, index):
        image_data = [read_image(self.lr_path[i]) / 255.0 for i in range(index, index + self.gof, 1)]
        image = torch.stack(image_data)
        label = read_image(self.hr_path[index + self.gof // 2]) / 255.0
        image = fix_context_change(image, self.threshold)
        return image, label
    
class CustomDatasetInf(Dataset):
    def __init__(self, lr_path, gof, threshold=0.3):
        self.gof = gof
        self.lr_path = lr_path
        self.threshold = threshold

    def __len__(self):
        return len(self.lr_path) - self.gof

    def __getitem__(self, index):
        image_data = [read_image(self.lr_path[i]) / 255.0 for i in range(index, index + self.gof, 1)]
        image = torch.stack(image_data)
        image = fix_context_change(image, self.threshold)
        return image


def fix_context_change(I, t): # Creating context change function for Training only
    """
    Input dimension (b,gof,c,h,w)
    t : Threshold
    """
    (gof, c, h, w) = I.size()
    ref = I[gof//2,:,:,:]
    ind = [j for j in range(gof) if(piq.ssim(I[j,:,:,:].unsqueeze(0), ref.unsqueeze(0), data_range=1.) < t and j != gof//2)]
    if(len(ind) > 0):
        if(len(list(filter(lambda x: x > gof//2, ind))) == len(ind)):
            for k in ind:
                I[k,:,:] = I[k-1,:,:]
        elif(len(list(filter(lambda x: x < gof//2, ind))) == len(ind)):
            for k in reversed(ind):
                I[k,:,:] = I[k+1,:,:]
    return I

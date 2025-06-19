import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features.eval()
        self.resize = resize

    def forward(self, input, target):
        loss = nn.functional.mse_loss(self.vgg(input), self.vgg(target))
        return loss

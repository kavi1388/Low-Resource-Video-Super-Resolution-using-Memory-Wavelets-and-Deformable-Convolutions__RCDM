import torch
import torch.nn as nn
from model.blocks import conv_layer, Blocks, ESA, pixelshuffle_block

class ConvNeXt(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, depths=[1, 3, 1, 1], dims=[72, 72, 64, 64], upscale_factor=4):
        super().__init__()
        self.conv1 = conv_layer(in_chans, dims[3], 3)
        self.downsample_layers = nn.ModuleList() 
        stem = conv_layer(dims[3], dims[0], kernel_size=3)
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = conv_layer(dims[i], dims[i + 1], kernel_size=3)
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Blocks(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
        self.esa = nn.ModuleList()
        for i in range(4):
            esa_layer = ESA(dims[i], nn.Conv2d)
            self.esa.append(esa_layer)
        self.conv2 = conv_layer(dims[3], dims[3], 3)
        self.upsample_block = pixelshuffle_block(dims[3], out_chans, upscale_factor)

    def forward(self, x):
        x = self.conv1(x)
        shortcut = x
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            x = self.esa[i](x)
        x = shortcut + x
        x = self.conv2(x)
        x = self.upsample_block(x)
        return x

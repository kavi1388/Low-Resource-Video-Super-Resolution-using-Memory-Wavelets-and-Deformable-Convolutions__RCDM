import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
from model.deform_conv3d import *
from model.convnext import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class UpsampleOneStep(nn.Sequential):
    
    
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape
    def forward(self, input):
        return input.view((-1,) + self.shape)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class RCDM(nn.Module):

    def __init__(self, num_channels, kernel_size, padding,
                 scale, activation, group_of_frames, num_of_patches,state=None):
        super(RCDM, self).__init__()

        print(
            f'num_channels, group_of_frames, kernel_size, padding, scale, activation,  {num_channels, group_of_frames, kernel_size, padding, scale, activation, num_of_patches}')
        self.num_of_patches = num_of_patches
        self.group_of_frames = group_of_frames
        self.scale = scale
        
        self.deformable_convolution1 = DeformConv3d((group_of_frames),
                                                  512, 
                                                  kernel_size = kernel_size[0], 
                                                  padding = padding[0])

#         Upsampling Layer
        self.up = nn.Upsample(scale_factor=(2,2), mode='bicubic', align_corners=True, recompute_scale_factor=True)

        # Wavelets
        self.dwt = DWTForward(J=1, mode='zero', wave='db1')

        # Add rest of the layers
    
        self.conv1 = nn.Conv3d(
            in_channels=group_of_frames, out_channels= 512, kernel_size=(1,1,1), padding=0)
        
#         self.batchnorm1 = nn.BatchNorm3d(256)
        
        self.conv11 = nn.Conv3d(
            in_channels=512+self.group_of_frames, out_channels= 1024, kernel_size=(1,1,1), padding=0)
        
#         self.batchnorm11 = nn.BatchNorm3d(512)
        
        self.conv_forDef = nn.Conv3d(
            in_channels=512, out_channels= 1024, kernel_size=(1,1,1), padding=0)
#         self.batchnorm_fordefconv = nn.BatchNorm3d(256)
        
        
        self.conv2 = nn.Conv3d(in_channels= 2560 +self.group_of_frames, out_channels=64, kernel_size=(1,1,1), padding=0)
        
#         self.batchnorm2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv2d(
            in_channels= 300, out_channels=32*num_channels,
            kernel_size=kernel_size, padding=padding)
#         self.batchnorm3 = nn.BatchNorm2d(32*num_channels)
        
        self.conv33 = nn.Conv2d(
            in_channels= 204, out_channels=32*num_channels,
            kernel_size=kernel_size, padding=padding)
#         self.batchnorm33 = nn.BatchNorm2d(32*num_channels)

#         self.space_to_depth = nn.PixelUnshuffle(scale)
#         self.up_block = nn.PixelShuffle(scale)
        self.prelu = nn.PReLU()
#         self.swin = SwinIR(upscale=scale, img_size=frame_size,
#                    window_size=4, img_range=1., depths=[6, 6, 6, 6],
#                    embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.upsample = UpsampleOneStep(scale, 32*num_channels, num_channels)
        self.alpha = torch.nn.Parameter(torch.rand(1))
        self.sr_convnext = ConvNeXt()
        


    def forward(self, X, state=None):
        
        lr = X
        df = self.deformable_convolution1(X)
        
        x = self.conv1(lr)
        x = self.prelu(x)
#         x = self.batchnorm1(x)
        
        x = torch.cat([x,lr],dim = 1)
        
        y = self.conv11(x)
        y = self.prelu(y)
#         y = self.batchnorm11(y)
        
        x = torch.cat([y,x],dim = 1)
        
        xdf = self.conv_forDef(df)
        xdf = self.prelu(xdf)
#         xdf = self.batchnorm1(xdf)
#         print(f'x {x.shape}')
        if state is None:
    
            state = torch.zeros(lr.shape[0],3,lr.shape[3]*4,lr.shape[4]*4).to(device)
#         x = torch.cat([state,x],dim = 1)
        y = []
#         for ind in range(self.group_of_frames):

        ind = self.group_of_frames//2
        Yl,Yh = self.dwt(lr[:,ind,:,:,:])
#         print(f' y1 {Yl.shape}')
        Ylu = self.up(Yl)
        Yh0u = self.up(Yh[0][:,0,:,:,:])
        Yh1u = self.up(Yh[0][:,1,:,:,:])
        Yh2u = self.up(Yh[0][:,2,:,:,:])
        y.append(Ylu)
        y.append(Yh0u)
        y.append(Yh1u)
        y.append(Yh2u)
        
        y = torch.stack(y, dim=1)

#         print(f'y {y.shape}')
#         
#         print(f'x {x.shape}')
            
        x = torch.cat([x,xdf],dim=1)
#         print(f'x {x.shape}')
        x = self.conv2(x)
        x = self.prelu(x)
#         x = self.batchnorm2(x)
        y = y[:,:,:,:45,:]
        x = torch.cat([y,x],dim = 1)
        
#         print(f'x {x.shape}')

        x = x.flatten(start_dim=1, end_dim=2)
#         x = x.reshape(x[0],x[1]*x[2],)
#         state = self.conv2(x)
#         print(f'x flatten {x.shape}')
        y = self.conv33(x)
        y = self.prelu(y)
#         y = self.batchnorm33(y)
        
        x = torch.cat([y,x],dim = 1)
        
        x = self.conv3(x)
        x = self.prelu(x)
#         x = self.batchnorm3(x)
        
        output = self.upsample(x)
        sr = self.sr_convnext(lr[:,self.group_of_frames//2,:,:,:])

#         bicubic = nn.functional.interpolate(lr[:,self.group_of_frames//2,:,:,:], scale_factor=self.scale, mode='bicubic', align_corners=False)

        output = torch.add(output,sr)
#         output = nn.Sigmoid()(output)
        op_updated = []
        for ind in range(lr.shape[0]):
            if ind < self.num_of_patches or ind==0:
                op_updated.append(torch.add(output[ind,:,:,:],torch.tanh(self.alpha*state[ind - self.num_of_patches,:,:,:])))
            else:
                op_updated.append(torch.add(output[ind,:,:,:],torch.tanh(self.alpha*output[ind-self.num_of_patches,:,:,:])))
#             print(f'op_updated {op_updated[-1].shape}')
        
        output = torch.stack(op_updated,dim = 0)
#         print(f'op {output.shape}')
        
        state = output
#         print(f'state {state.shape}')
    
        output = nn.Sigmoid()(output)
    
#         output = motion_deblur(output)
        
#         print(f"output.shape {output.shape}")
        
        return output, state


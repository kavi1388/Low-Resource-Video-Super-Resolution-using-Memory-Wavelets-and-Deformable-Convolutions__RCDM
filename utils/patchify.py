import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, patch_size=(45, 80)):
        super().__init__()
        self.p = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.unfold(x)
        a = x.view(bs, c, self.p[0], self.p[1], -1).permute(0, 4, 1, 2, 3)
        return a

class Unpatchify(nn.Module):
    def __init__(self, patch_size=(180, 320), image_size=(720, 1280)):
        super().__init__()
        self.p = patch_size
        self.image_size = image_size
        self.fold = nn.Fold(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, patches):
        # patches -> (B, no_of_patches, C, pH, pW)
        B, no_of_patches, C, pH, pW = patches.shape

        # Flatten patches into a format suitable for nn.Fold
        patches = patches.permute(0, 2, 3, 4, 1).reshape(B, -1, no_of_patches)

        # Fold patches back to the original image
        image = self.fold(patches)

        # Normalize overlap if patches overlap
        overlap_count = self.fold(torch.ones_like(patches))
        image /= overlap_count

        return image


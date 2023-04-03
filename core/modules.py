import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResidualConv(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv2(self.relu(self.conv1(x)))
        return x + res
        
class ResidualConv(nn.Module):
    def __init__(self, filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.conv2(self.relu(self.conv1(x)))
        return x + res

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch=64, blocks=30):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1),
                                  nn.LeakyReLU(0.1))
        self.res_block = nn.Sequential(*[ResidualConv(out_ch) for _ in range(blocks)])

    def forward(self, x):
        x = self.conv(x)
        return self.res_block(x)
    
class PixelShufflePack(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor):
        super().__init__()
        self.upconv = nn.Conv2d(in_ch, out_ch * upscale_factor * upscale_factor, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.lrelu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.upconv(x)
        return self.lrelu(self.pixel_shuffle(x))
    
class CleaningModule(nn.Module):
    def __init__(self, mid_ch, blocks):
        super().__init__()
        self.resblock = ResidualBlock(3, mid_ch, blocks)
        self.conv = nn.Conv2d(mid_ch, 3, 3, 1, 1, bias=True)
    
    def forward(self, x):
        x = self.resblock(x)
        return self.conv(x)
    
class SpectralConv(nn.Module):
  def __init__(self, in_ch, out_ch, ks=3, stride=1, pad=1):
    super().__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=False)
  
  def forward(self, x):
    x = self.conv(x)
    return spectral_norm(x)
        
        
    

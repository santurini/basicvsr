import sys
sys.path.append('./')

import torch
import torch.nn as nn
from models.basicvsr import BasicVSR
from core.modules import CleaningModule

class RealBasicVSR(nn.Module):
    def __init__(self, mid_channels=64, cleaning_blocks=20, blocks=20, upscale=4, threshold=1., is_mirror=False):
        super().__init__()
        self.name = 'Real Basic VSR'
        self.scale = upscale
        self.cleaner = CleaningModule(mid_channels, cleaning_blocks)
        self.basicvsr = BasicVSR(mid_channels, blocks, upscale, is_mirror)
        self.threshold = threshold

    def forward(self, lqs):
        n, t, c, h, w = lqs.size()
        for _ in range(3):  # at most 3 cleaning, determined empirically
            lqs = lqs.view(-1, c, h, w)
            residues = self.cleaner(lqs)
            lqs = (lqs + residues).view(n, t, c, h, w)
            if torch.mean(torch.abs(residues)) < self.threshold: 
                break
        outputs = self.basicvsr(lqs)
        return outputs, lqs       
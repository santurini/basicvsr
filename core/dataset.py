import os
import torch
import random
from PIL import Image
from pathlib import Path
from core.augmentations import RandomCrop
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_tensor

class VSR(torch.utils.data.Dataset):
    def __init__(self, path, crop=64, seq=30, mirror=True, scale=4, split='train', size=0.8):
        if mirror:
            assert seq%2==0, "Sequence length must be divisble by 2 if mirroring is used!"
            self.seq = seq // 2
            self.mirror = mirror
        else:
            self.seq = seq
            self.mirror = mirror
        self.scale = scale
        self.lr_path = list(sorted(Path(os.path.join(path,'LR')).glob('*')))
        self.hr_path = list(sorted(Path(os.path.join(path,'HR')).glob('*')))
        split_point = int(len(self.hr_path)*size)
        if split=='train':
            self.lr_path = self.lr_path[:split_point]
            self.hr_path = self.hr_path[:split_point]
        else:
            self.lr_path = self.lr_path[split_point:]
            self.hr_path = self.hr_path[split_point:]           
        
    def __len__(self):
        return len(self.lr_path)
    
    def __getitem__(self, idx):  
        assert self.lr_path[idx].name == self.hr_path[idx].name
        lr_video = list(sorted(self.lr_path[idx].glob('*')))    
        hr_video = list(sorted(self.hr_path[idx].glob('*')))  
        rnd = random.randint(0, len(lr_video) - self.seq)
        lr_batch, hr_batch = self.get_frames(lr_video, hr_video, rnd)
        lr_batch, hr_batch = self.random_crop(lr_batch, hr_batch)
        return lr_batch, hr_batch
    
    def load_img(self, path):
        return to_tensor(Image.open(path))

    def get_frames(self, lr_video, hr_video, rnd):
        lr_video = torch.stack([self.load_img(i) for i in lr_video[rnd:rnd+self.seq]])
        hr_video = torch.stack([self.load_img(i) for i in hr_video[rnd:rnd+self.seq]])
        if self.mirror:
            lr_video = torch.concat((lr_video, lr_video.flip(1)), 1),
            hr_video = torch.concat((lr_video, hr_video.flip(1)), 1)
        return lr_video, hr_video
    
    def random_crop(self, lr_video, hr_video):
        height, width = self.pair(self.crop)
        x = random.randint(0, lr_video.shape[-1] - width)
        y = random.randint(0, lr_video.shape[-2] - height)
        lr_video = lr_video[:, :, :, y:y+height, x:x+width]
        hr_video = hr_video[:, :, :, y*self.scale:(y+height)*self.scale, x*self.scale:(x+width)*self.scale]
        return lr_video, hr_video
        
    def pair(self, t):
        return t if isinstance(t, tuple) else (t, t)
    
class RealVSR(torch.utils.data.Dataset):
    def __init__(self, path, crop=256, seq=15, scale=4, split='train', size=0.8, augment=None, is_mirror=False):
        if is_mirror:
            assert seq%2==0, "Sequence length must be divisble by 2 if mirroring is used!"
            self.seq = seq // 2
            self.is_mirror = is_mirror
        else:
            self.seq = seq
            self.is_mirror = is_mirror
        self.scale = scale
        self.crop = crop
        self.random_crop = RandomCrop(crop)
        self.augment = augment
        self.path = list(sorted(Path(path).glob('*')))
        split_point = int(len(self.path)*size)
        if split=='train':
            self.path = self.path[:split_point]
        else:
            self.path = self.path[split_point:]
        
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):  
        video = list(sorted(self.path[idx].glob('*')))    
        rnd = random.randint(0, len(video) - self.seq)
        batch = self.get_frames(video, rnd)
        assert self.crop <= batch.shape[-1] & self.crop <= batch.shape[-2]
        hr_batch = self.random_crop(batch)
        if self.is_mirror:
            hr_batch = self.mirror(hr_batch)
        if self.augment!= None:
            lr_batch = self.augment(hr_batch)
        else:
            lr_batch = F.resize(hr_batch, (self.crop//self.scale, self.crop//self.scale))      
              
        return lr_batch, hr_batch
    
    def load_img(self, path):
        return to_tensor(Image.open(path))

    def get_frames(self, video, rnd):
        video = [self.load_img(i) for i in video[rnd:rnd+self.seq]]
        return torch.stack(video)
    
    def mirror(self, batch):
        flipped = batch.flip(0)
        extended = torch.concat((batch, flipped), 0)
        return extended
    

        

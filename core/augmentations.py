import io
import av
import torch
import random
import torch.nn as nn
from PIL import Image
from pathlib import Path
from einops.layers.torch import Rearrange
from torchvision.transforms import Resize
import torchvision.transforms.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image

class RandomCrop(nn.Module):
  def __init__(self, crop_size):
    super().__init__()
    self.crop_size = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)

  def forward(self, video):
    height, width = self.crop_size
    x = random.randint(0, video.shape[-1] - width)
    y = random.randint(0, video.shape[-2] - height)
    video = video[..., y:y+height, x:x+width]
    return video

class RandomGaussianBlur(nn.Module):
    def __init__(self, kernel_size=(3, 5, 7), sigma=(0.2, 2.0)):
        super().__init__()
        self.kernel_size = random.choice(kernel_size) if isinstance(kernel_size, tuple) else kernel_size
        self.sigma = torch.empty(1).uniform_(sigma[0], sigma[1]).item() if isinstance(sigma, tuple) else sigma

    def forward(self, img):
        if len(img.shape)==4:
            out = torch.stack([F.gaussian_blur(i, self.kernel_size, self.sigma) for i in img])
        elif len(img.shape)==3:
            out = F.gaussian_blur(img, self.kernel_size, self.sigma)
        else:
            raise ValueError('Is this even an image bro????')
        return out

class RandomResize(nn.Module):
  def __init__(self, scale):
    super().__init__()
    self.scale = torch.empty(1).uniform_(scale[0], scale[1]).item() if isinstance(scale, tuple) else scale
  
  def forward(self, x):
    new_width = (int(x.shape[-1]*self.scale) // 2) * 2
    new_height = (int(x.shape[-2]*self.scale) // 2) * 2
    if len(x.shape)==4:
        out = torch.stack([F.resize(i, (new_height, new_width)) for i in x])
    elif len(x.shape)==3:
        out = F.resize(x, (new_height, new_width))
    else:
        raise ValueError('Is this even an image bro????')
    return out

class RandomJPEGCompression(torch.nn.Module):
    def __init__(self, q):
        super().__init__()
        self.q = random.randint(q[0], q[1]) if isinstance(q, tuple) else q
        
    def forward(self, in_batch):
        return self.torch_batch_add_compression(in_batch.detach().clamp(0, 1), self.q).type_as(in_batch)
    
    def pil_add_compression(self, pil_img: Image.Image, q: int) -> Image.Image:
        # BytesIO: just like an opened file, but in memory
        with io.BytesIO() as buffer:
            # do the actual compression
            pil_img.save(buffer, format='JPEG', quality=q)
            buffer.seek(0)
            with Image.open(buffer) as compressed_img:
                compressed_img.load()  # keep image in memory after exiting the `with` block
                return compressed_img
            
    def torch_add_compression(self, in_tensor: torch.Tensor, q: int) -> torch.Tensor:
        pil_img = to_pil_image(in_tensor)
        compressed_img = self.pil_add_compression(pil_img, q=q)
        return to_tensor(compressed_img).type_as(in_tensor)

    def torch_batch_add_compression(self, in_batch: torch.Tensor, q: int) -> torch.Tensor:
        return torch.stack([self.torch_add_compression(elem, q) for elem in in_batch])
    
class RandomVideoCompression(nn.Module):
    def __init__(self, codec, bitrate):
        super().__init__()
        self.codec = random.choice(codec) if isinstance(codec, tuple) else codec
        self.bitrate = random.choice(bitrate) if isinstance(bitrate, tuple) else bitrate
        
    def forward(self, video):
        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            stream = container.add_stream(self.codec, rate=1)
            stream.height = video[0].shape[-2]
            stream.width = video[0].shape[-1]
            stream.pix_fmt = 'yuv420p'
            stream.bit_rate = self.bitrate
            for frame in video:
              frame = av.VideoFrame.from_image(to_pil_image(frame))
              frame.pict_type = 'NONE'    
              for packet in stream.encode(frame):
                  container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
          if container.streams.video:
              for frame in container.decode(**{'video': 0}):
                  outputs.append(to_tensor(frame.to_image()))
                  
        return torch.stack(outputs)
    
class RandAugment(nn.Module):
  def __init__(self, crop_size, scale) -> None:
    super().__init__()
    kernel_size = (7, 9, 11, 13, 15, 17, 19, 21)
    sigma = (0.2, 3)
    resize_scale = (0.15, 1.5)
    quality = (30, 80)
    codec = 'h264'
    bitrate = 5e3
    resize_scale_2 = (0.3, 1.2)
    self.aug = nn.Sequential(RandomGaussianBlur(kernel_size, sigma),
                        RandomResize(resize_scale),
                        RandomJPEGCompression(quality),
                        RandomVideoCompression(codec, bitrate),
                        RandomGaussianBlur(kernel_size, sigma),
                        RandomResize(resize_scale_2),
                        RandomJPEGCompression(quality),
                        RandomGaussianBlur(kernel_size, sigma),
                        RandomVideoCompression(codec, bitrate),
                        Resize((crop_size//scale, crop_size//scale))).to('cuda')
  
  def forward(self, x):
    lr = self.aug(x)
    return lr

class AugmentPipeline(nn.Module):
  def __init__(self, aug) -> None:
    super().__init__()
    self.aug = aug
  
  def forward(self, x):
    lr = self.aug(x)
    return lr
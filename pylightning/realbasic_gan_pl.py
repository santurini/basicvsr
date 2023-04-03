import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from core.losses import CharbonierLoss, AdversarialLoss, PerceptualLoss
from models.unet import UNetDiscriminator
from piqa import PSNR

class GanRealVSR(pl.LightningModule):
    def __init__(self, generator, discriminator, optimizer, scheduler):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CharbonierLoss()
        self.perceptual = PerceptualLoss()
        self.adversarial = AdversarialLoss()   
        self.psnr = PSNR()

    def forward(self, x):
        sr, lq = self.generator(x)
        return sr, lq
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            return self.generator_step(batch, batch_idx)
        elif optimizer_idx == 1:
            return self.discriminator_step(batch)

    def generator_step(self, batch, batch_idx):
        lr, hr = batch
        b, t, c, h, w = lr.shape
        sr, lq = self.forward(lr)
        self.discriminator.requires_grad_(False)
        pixel_loss = self.criterion(sr, hr) + self.criterion(lq, F.resize(hr, (h, w)))
        perceptual_loss = self.perceptual(sr, hr)
        disc_sr = self.discriminator(sr)
        disc_fake_loss = self.adversarial(disc_sr, 1, False)
        loss = pixel_loss + perceptual_loss + disc_fake_loss
        psnr = self.psnr(torch.clamp(sr, 0, 1), hr)
        self.log('loss', loss, prog_bar=True)
        self.log('psnr', psnr, prog_bar=True)
        if batch_idx%1000 == 0:
          self.log_images(sr, hr)
        return loss

    def discriminator_step(self, batch):
        lr, hr = batch
        b, t, c, h, w = lr.shape
        sr, lq = self.forward(lr)
        self.discriminator.requires_grad_(True)
        disc_hr = self.discriminator(hr)
        disc_true_loss = self.adversarial(disc_hr, 1, True)
        disc_sr = self.discriminator(sr.detach())
        disc_fake_loss = self.adversarial(disc_sr, 0, True)
        loss = disc_fake_loss + disc_true_loss
        return loss

    def configure_optimizers(self):
        if self.scheduler == None:
            return [self.optimizer[0], self.optimizer[1]], []
        else:
            g_sched = self.scheduler[0]
            d_sched = self.scheduler[1]
            return [self.optimizer[0], self.optimizer[1]], [g_sched, d_sched]

    def log_images(self, lr, lq, sr, hr, res_hr):
        b, t, c, h, w = lr.shape; t_log = 5
        hr = hr[0][:t_log]
        sr = sr[0][:t_log].clamp(0, 1)
        psnr = ['PSNR: ' + str(self.psnr(i, j).detach().cpu().numpy().round(2)) for i, j in zip(sr, hr)]
        self.logger.log_image(key='Ground Truths', images=[i for i in hr], caption=[f'gt_frame_{i+1}' for i in range(t_log)])
        self.logger.log_image(key='Predicted Images', images=[i for i in sr], caption=psnr)
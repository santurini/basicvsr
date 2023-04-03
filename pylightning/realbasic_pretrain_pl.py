import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from core.losses import CharbonierLoss
from piqa import PSNR

class PretrainRealVSR(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CharbonierLoss()
        self.psnr = PSNR()
        
    def forward(self, x):
        sr, lq = self.model(x)
        return sr, lq
    
    def shared_step(self, batch, batch_idx, stage):
        lr, hr = batch
        b, t, c, h, w = hr.shape
        sr, lq = self.forward(lr)
        resized_hr = F.interpolate(hr.view(-1, c, h, w), scale_factor=1/self.model.scale, mode='area')
        loss = self.criterion(lq, resized_hr) + self.criterion(sr, hr)
        psnr = self.psnr(sr.clamp(0, 1), hr.clamp(0, 1))
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/psnr', psnr, prog_bar=True)
        if ((batch_idx%1000 == 0) & (stage=='val')): 
          self.log_images(lr, sr, hr)
        return {"loss": loss, "psnr": psnr}
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train") 

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def log_images(self, lr, sr, hr):
        t_log = 5
        lr = lr[0][:t_log]
        hr = hr[0][:t_log].clamp(0, 1)
        sr = sr[0][:t_log].clamp(0, 1)
        psnr = ['PSNR: ' + str(self.psnr(i, j).detach().cpu().numpy().round(2)) for i, j in zip(sr, hr)]
        self.logger.log_image(key='Ground Truths', images=[i for i in hr], caption=[f'gt_frame_{i+1}' for i in range(t_log)])
        self.logger.log_image(key='Predicted Images', images=[i for i in sr], caption=psnr)
        self.logger.log_image(key='Input Images', images=[i for i in lr], caption=[f'input_frame_{i+1}' for i in range(t_log)])

        

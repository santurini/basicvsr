import sys
sys.path.append('./')

import torch
import pytorch_lightning as pl
from core.losses import CharbonierLoss
from piqa import PSNR

class LightningVSR(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        self.model = model 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = CharbonierLoss()
        self.psnr = PSNR()
        
    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, batch_idx, stage):
        lr, hr = batch
        sr = self.forward(lr) 
        loss = self.criterion(sr, hr)
        psnr = self.psnr(torch.clamp(sr, 0, 1), hr)
        self.log(f'{stage}/loss', loss, prog_bar=True)
        self.log(f'{stage}/psnr', psnr, prog_bar=True)
        if ((batch_idx%50 == 0) & (stage=='val')): 
          self.log_images(lr, hr, sr)
        return {"loss": loss, "psnr": psnr}
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "train") 

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = self.scheduler
        return [optimizer], [scheduler]

    def log_images(self, lr, hr, sr):
        _, t, _, _, _ = lr.shape
        lr = [i for i in lr[0]]
        hr = [i for i in hr[0]]
        sr = [torch.clamp(i, 0, 1) for i in sr[0]]
        psnr = ['PSNR: ' + str(self.psnr(i, j).detach().cpu().numpy().round(2)) for i, j in zip(sr, hr)]
        self.logger.log_image(key='Ground Truths', images=hr, caption=[f'gt_frame_{i+1}' for i in range(t)])
        self.logger.log_image(key='Predicted Images', images=sr, caption=psnr)
        self.logger.log_image(key='Input Images', images=lr, caption=[f'input_frame_{i+1}' for i in range(t)])
        

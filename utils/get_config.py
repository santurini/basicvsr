import sys
sys.path.append('./')

import json
import wandb
import torch
import torch.nn as nn
from core.augmentations import *
from core.dataset import RealVSR
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.transforms import Resize
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from deepspeed.ops.adam import DeepSpeedCPUAdam
from lightning.realbasic_pretrain_pl import *
from lightning.realbasic_gan_pl import *


def train(cfg):
    print('\nPREPARING DATALOADERS ...')
    tr_dl, val_dl = get_loaders(cfg)
    print('\nDEFINING LIGHTNING MODULE ...')
    pl_module = get_model(cfg)
    print('\INITIALIZING TRAINER ...')
    trainer = get_trainer(cfg)
    print("\nSTARTING TRAINING FOR", args.epochs, "EPOCHS\n")
    trainer.fit(pl_module, tr_dl, val_dl, ckpt_path=cfg["trainer"]["ckpt_path"])
    print("\nTRAINING COMPLETED SUCCESFULLY!")
    return

def get_trainer(cfg):
    cfg = cfg["trainer"]
    ckpt = ModelCheckpoint(dirpath='./checkpoint', monitor='val/loss', save_last=True)  
    logger = WandbLogger(project='video-super-resolution')
    trainer = Trainer(callbacks=[ckpt], 
                  accelerator=cfg["device"], 
                  max_epochs=cfg["epochs"],
                  logger=logger,
                  strategy=cfg["strategy"], 
                  devices=cfg["n_devices"],
                  precision=cfg["precision"])
    return trainer

def get_model(cfg):
    cfg = cfg["model"]
    if cfg["name"]=="realbasicvsr":
        model = RealBasicVSR(*cfg["args"])
        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)
        pl_module = PretrainRealVSR(model, optimizer, scheduler)       
    elif cfg["name"]=="ganbasicvsr":
        generator = RealBasicVSR(*cfg["args"])
        discriminator = UnetDiscriminator()
        optimizer = (get_optimizer(cfg, generator), get_optimizer(cfg, discriminator))
        scheduler = (get_scheduler(cfg, opt_d), get_scheduler(cfg, opt_g))
        pl_module = FinetuneRealVSR(generator, discriminator, optimizer, scheduler)
    return pl_module
        
def get_optimizer(cfg, model):
    cfg = cfg["optimizer"]
    if cfg["name"]=="Adam":
        optimizer = Adam(model.parameters(), lr=cfg["lr"])
    elif cfg["name"]=="AdamW":
        optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    elif cfg["name"]=="DeepSpeedCPUAdam":
        optimizer = DeepSpeedCPUAdam(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError("Invalid Argument")
    return optimizer

def get_scheduler(cfg, optimizer):
    cfg = cfg["scheduler"]
    if cfg["name"]=="cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T"], eta_min=cfg["min_lr"])
    elif cfg["name"]=="none":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["T"], eta_min=optimizer.lr)
    return scheduler

def get_loaders(cfg):
    cfg = cfg["dataloader"]
    is_mirror = cfg["model"]["args"]["mirror"]
    augment_pipeline = get_augmentation(cfg["augmentation"])
    tr_ds = RealVSR(cfg["path"], cfg["crop_size"], cfg["seq"], cfg["scale"], 
                 "train", cfg["size"], augment_pipeline, is_mirror)
    val_ds = RealVSR(cfg["path"], cfg["crop_size"], cfg["seq"], cfg["scale"], 
                 "val", cfg["size"], None, is_mirror)
    tr_dl = DataLoader(tr_ds,  cfg["batch_size"], True, num_workers=cfg["cpu_w"])
    val_dl = DataLoader(tr_ds,  cfg["batch_size"], False, num_workers=cfg["cpu_w"])
    return tr_dl, val_dl
    
def get_augmentation(cfg):
    aug = parse_aumentation(cfg)
    return AugmentPipeline(aug)

def parse_aumentation(cfg):   
    for key in cfg.keys():
        aug = key.split['_'][0]
        if aug=='gaussian':
            kernel, sigma = cfg[key].values()
            t = RandomGaussianBlur(kernel, sigma)
            pipeline.append(t)
        if aug=='resize':
            scale = cfg[key].values()
            t = RandomResize(scale)
            pipeline.append(t)
        if aug=='jpeg':
            quality = cfg[key].values()
            t = RandomJPEGCompression(quality)
            pipeline.append(t)
        if aug=='videocompression':
            codec, bitrate = cfg[key].values()
            t = RandomVideoCompression(codec, bitrate)
            pipeline.append(t)
        if aug=='crop':
            height, width = cfg[key].values()
            t = RandomCrop((height, width))
            pipeline.append(t)
        if aug=='fixedresize':
            height, width = cfg[key].values()
            t = Resize((height, width))
            pipeline.append(t)  
    return nn.Sequential(*pipeline)
    
    
    
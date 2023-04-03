import sys 
sys.path.append('./') 

import torch
import wandb
import argparse
from core.dataset import VSRDataset
from models.basicvsr import BasicVSR
from models.dsvsr import DSVSR
from models.realbasicvsr import RealBasicVSR
from lightning.lightning_basic import LightningVSR
from lightning.lightning_real import LightningRealVSR
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=False, default='gpu', help='training device (cpu/gpu)')
parser.add_argument('--n_devices', type=int, required=False, default=None, help='number of devices')
parser.add_argument('--cpu_w', type=int, required=False, default=0, help='number of cpu workers in dataloader')
parser.add_argument('--path', type=str, required=True, help='input folder path')
parser.add_argument('--seq', type=int, required=False, default=7, help='sequence length')
parser.add_argument('--split', type=float, required=False, default=0.8, help='train/val split')
parser.add_argument('--bs', type=int, required=False, default=2, help='batch size')
parser.add_argument('--epochs', type=int, required=False, default=300, help='train epochs')
parser.add_argument('--lr', type=float, required=False, default=2e-4, help='learning rate')
parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='path to ckeckpoint')
parser.add_argument('--mid_ch', type=int, required=False, default=64, help='mid channels')
parser.add_argument('--blocks', type=int, required=False, default=20, help='residual blocks')
parser.add_argument('--cl_blocks', type=int, required=False, default=20, help='cleaning module residual blocks')
parser.add_argument('--model', type=str, required=False, default='real', help='ds (depth-separable) / bs (basic) / real')
parser.add_argument('--scale', type=int, required=False, default=2, help='upscale factor')
parser.add_argument('--threshold', type=float, required=False, default=1.0, help='cleaning threshold')
parser.add_argument('--strategy', type=str, required=False, default='deepspeed_stage_3_offload', help='training strategy, check lighnting docs')
parser.add_argument('--precision', type=int, required=False, default=16, help='model precision , usually 16 or 32')
parser.add_argument('--clean_img', action='store true', help='log flag for cleaned images')
args = parser.parse_args()

wandb.login(key='191e81893a41f570331354ae4c2aa8e99a4bba48')    
logger = WandbLogger(project='video-super-resolution')

print('\nPREPARING DATALOADERS:')
tr_ds = VSRDataset(args.path, args.seq, 'train', args.split)
val_ds = VSRDataset(args.path, args.seq, 'val', args.split)
tr_dl = DataLoader(tr_ds,  args.bs, False, num_workers=args.cpu_w)
val_dl = DataLoader(val_ds,  args.bs, False, num_workers=args.cpu_w)

if args.model=='ds':
    model = DSVSR(args.mid_ch, args.blocks, args.scale) 
if args.model=='basic':
    model = BasicVSR(args.mid_ch, args.blocks, args.scale)
if args.model=='real':
    model = RealBasicVSR(args.mid_ch, args.cl_blocks, args.blocks, args.scale, args.threshold)

print("\nLOADING MODEL CONFIGURATION:", model.name, "with", args.blocks, "blocks")
print('\nNUMBER OF TRAINABLE PARAMETERS:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr) if 'deepspeed' in args.strategy else Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-07)
lr_monitor = LearningRateMonitor('step')
ckpt = ModelCheckpoint(dirpath='./checkpoint',
                       filename='./ckpt-{val_psnr:.2f}',
                       monitor='val/loss', 
                       save_last=True)

print('\nPREPARING LIGHTNING MODULE . . .')
if args.strategy=='none':
    strategy = None
if 'deepspeed' in args.strategy:
    strategy =  DeepSpeedStrategy(stage=3,offload_optimizer=True,offload_parameters=True,allgather_bucket_size=1e5,reduce_bucket_size=1e5)

if args.model=='real': 
    pl_module = LightningRealVSR(model, optimizer, scheduler, args.clean_img)
else:
    pl_module = LightningVSR(model, optimizer, scheduler)
    
trainer = Trainer(callbacks=[ckpt, lr_monitor], 
                  accelerator=args.device, 
                  max_epochs=args.epochs,
                  accumulate_grad_batches=32//args.bs,
                  logger=logger,
                  strategy=strategy, 
                  precision=args.precision,
                  devices=args.n_devices)

print("\nSTARTING TRAINING FOR", args.epochs, "EPOCHS\n")
trainer.fit(pl_module, tr_dl, val_dl, ckpt_path=args.ckpt_path)



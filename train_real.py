import sys 
import warnings
sys.path.append('./') 
warnings.filterwarnings("ignore")

import wandb
import argparse
from core.dataset import RealVSR
from core.augmentations import RandAugment
from models.realbasicvsr import RealBasicVSR
from pylightning.realbasic_pretrain_pl import PretrainRealVSR
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, required=False, default='gpu', help='training device (cpu/gpu)')
parser.add_argument('--n_devices', type=int, required=False, default=1, help='number of devices')
parser.add_argument('--cpu_w', type=int, required=False, default=0, help='number of cpu workers in dataloader')
parser.add_argument('--path', type=str, required=True, help='input folder path')
parser.add_argument('--seq', type=int, required=False, default=15, help='sequence length')
parser.add_argument('--mirror', action="store_true", help="mirror or not sequence")
parser.add_argument('--split', type=float, required=False, default=0.9, help='train/val split')
parser.add_argument('--bs', type=int, required=False, default=2, help='batch size')
parser.add_argument('--epochs', type=int, required=False, default=1500, help='train epochs')
parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
parser.add_argument('--scheduler', action="store_true", help='use or not a scheduler')
parser.add_argument('--ckpt_path', type=str, required=False, default=None, help='path to ckeckpoint')
parser.add_argument('--crop_size', type=int, required=False, default=256, help='crop_size')
parser.add_argument('--mid_ch', type=int, required=False, default=64, help='mid channels')
parser.add_argument('--blocks', type=int, required=False, default=20, help='residual blocks')
parser.add_argument('--cl_blocks', type=int, required=False, default=20, help='cleaning module residual blocks')
parser.add_argument('--model', type=str, required=False, default='pretrain', help='pretrain - fine')
parser.add_argument('--scale', type=int, required=False, default=4, help='upscale factor')
parser.add_argument('--augment', action="store_true", help='use or not augmentation')
parser.add_argument('--threshold', type=float, required=False, default=1.0, help='cleaning threshold')
parser.add_argument('--strategy', type=str, required=False, default='deepspeed_stage_3_offload', help='training strategy, check lighnting docs')
parser.add_argument('--precision', type=str, required=False, default='16-mixed', help='model precision , usually 16 or 32')
args = parser.parse_args()

wandb.login(key='191e81893a41f570331354ae4c2aa8e99a4bba48')    
logger = WandbLogger(project='video-super-resolution')

print('\nPREPARING DATALOADERS')
augmentation = RandAugment(args.crop_size, args.scale) if args.augment else None
tr_ds = RealVSR(args.path, args.crop_size, args.seq, args.scale, 'train', args.split, augmentation, args.mirror)
val_ds = RealVSR(args.path, args.crop_size, args.seq, args.scale, 'val', args.split, None, args.mirror)
tr_dl = DataLoader(tr_ds,  args.bs, True, num_workers=args.cpu_w)
val_dl = DataLoader(val_ds,  args.bs, False, num_workers=args.cpu_w)

model = RealBasicVSR(args.mid_ch, args.cl_blocks, args.blocks, args.scale, args.threshold, args.mirror)

print("\nLOADING MODEL CONFIGURATION:", model.name, "with", args.blocks, "blocks")
print('\nNUMBER OF TRAINABLE PARAMETERS:', sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = DeepSpeedCPUAdam(model.parameters(), lr=args.lr) if 'deepspeed' in args.strategy else Adam(model.parameters(), lr=args.lr)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-07) if args.scheduler else CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr)
ckpt = ModelCheckpoint(dirpath='./checkpoint',
                       monitor='val/loss', 
                       save_last=True)

print('\nPREPARING LIGHTNING MODULE . . .')

if 'deepspeed' in args.strategy:
    strategy =  DeepSpeedStrategy(stage=3,
                                  offload_optimizer=True,
                                  offload_parameters=True,
                                  allgather_bucket_size=1e5,
                                  reduce_bucket_size=1e5)
    

else:
    strategy = None

if args.model=='pretrain': 
    pl_module = PretrainRealVSR(model, optimizer, scheduler)
elif args.model=='fine':
    pl_module = FinetuneRealVSR(model, discriminator, optimizer, scheduler)
    
trainer = Trainer(callbacks=[ckpt], 
                  accelerator=args.device, 
                  max_epochs=args.epochs,
                  logger=logger,
                  strategy=strategy, 
                  precision=args.precision,
                  devices=args.n_devices)

print("\nSTARTING TRAINING FOR", args.epochs, "EPOCHS\n")
trainer.fit(pl_module, tr_dl, val_dl, ckpt_path=args.ckpt_path)
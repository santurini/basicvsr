import sys
sys.path.append('./')

from utils.get_config import train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=False, default='./configs/realbasic_pretrained_deepspeed.json', help='configuration file path')
args = parser.parse_args()

cfg_file = json.load(open(args.path))
train(cfg_file)
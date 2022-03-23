from ast import parse
from cProfile import run
import pytorch_lightning as pl
from argparse import ArgumentParser
from train import Model, run_test

parser = ArgumentParser()
parser.add_argument('--ckpt')
parser.add_argument('--data_dir', default='Dataset/')
parser.add_argument('--out', default='submit.csv')
# parser.add_argument('--cfg')
args = parser.parse_args()
# model = Model.load_from_checkpoint(args.ckpt)
run_test(args.ckpt, args.data_dir, args.out, 'cuda')
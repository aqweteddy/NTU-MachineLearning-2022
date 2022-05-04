import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
import wandb
from dataset import QADataset
from trainer import QATrainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from sklearn.model_selection import KFold

seed_everything(8787)
parser = ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--train_data', type=str, default='data/hw7_train.json')
parser.add_argument('--val_data', type=str, default='data/hw7_dev.json')
parser.add_argument('--exp_name', type=str, default='QA')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--maxlen', type=int, default=512)
parser.add_argument('--mask_prob', type=float, default=0)
parser.add_argument('--random_swap', action='store_true', default=False)
parser.add_argument('--to_cn', action='store_true', default=False)
parser = QATrainer.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

def run_train(train_loader: data.DataLoader, val_loader: data.DataLoader, postfix:str=''):
    args.warmup_steps = int(len(train_loader) * args.warmup_epochs / args.accumulate_grad_batches)
    args.num_train_steps = int(len(train_loader) * args.max_epochs  / args.accumulate_grad_batches)
    model = QATrainer(args)
    es = EarlyStopping(monitor='val/EM', patience=5, mode='max')
    mc = ModelCheckpoint(monitor='val/EM', save_last=False, save_top_k=1, mode='max')

    lrm = LearningRateMonitor('step')
    logger = WandbLogger(name=f'{args.exp_name}_{postfix}',
                                                            project='ML_hw7')
    trainer = Trainer.from_argparse_args(args,
                                        callbacks=[mc, lrm, es],
                                        logger=logger)
    trainer.fit(model, train_loader, val_loader)
    logger.experiment.finish()
    logger.finalize('finished')

train1_set = QADataset(args.train_data,
                        args.pretrained, args.maxlen,
                        'train', to_cn=args.to_cn, mask_prob=args.mask_prob, random_swap=args.random_swap)
train2_set = QADataset(args.val_data,
                    args.pretrained, args.maxlen,
                    'train', to_cn=args.to_cn, mask_prob=args.mask_prob, random_swap=args.random_swap)
tot_set = data.ConcatDataset([train1_set, train2_set])
kfold = KFold(args.kfold, shuffle=True)

for idx, (train_idx, val_idx) in enumerate(kfold.split(tot_set,)):
    train_set = data.Subset(tot_set, train_idx)
    val_set = data.Subset(tot_set, val_idx)
    for ds in val_set.dataset.datasets:
        ds.mode = 'val'
    
    train_loader = data.DataLoader(train_set,  args.batch_size, shuffle=True, num_workers=5)
    val_loader = data.DataLoader(val_set,  args.batch_size, num_workers=5)
    run_train(train_loader, val_loader, postfix=f'fold_{idx}')
    for ds in val_set.dataset.datasets:
        ds.mode = 'train'
    del train_loader, val_loader

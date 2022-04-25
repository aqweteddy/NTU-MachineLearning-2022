import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from dataset import QADataset
from trainer import QATrainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


seed_everything(8877)
parser = ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--train_data', type=str, default='data/hw7_train.json')
parser.add_argument('--val_data', type=str, default='data/hw7_dev.json')
parser.add_argument('--exp_name', type=str, default='QA')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--maxlen', type=int, default=512)
parser.add_argument('--to_cn', action='store_true', default=False)
parser = QATrainer.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid

train_loader = QADataset.dataloader(args.train_data,
                                    args.pretrained, args.maxlen,
                                     'train', args.batch_size, to_cn=args.to_cn)
val_loader = QADataset.dataloader(args.val_data, args.pretrained,
                                    args.maxlen,  'valid', args.batch_size, to_cn=args.to_cn)
args.warmup_steps = int(len(train_loader) * args.warmup_epochs / args.accumulate_grad_batches)
args.num_train_steps = int(len(train_loader) * args.max_epochs  / args.accumulate_grad_batches)
model = QATrainer(args)
es = EarlyStopping(monitor='val/EM', patience=5, mode='max')
mc = ModelCheckpoint(monitor='val/EM', save_last=False, save_top_k=1, mode='max')


lrm = LearningRateMonitor('step')

trainer = Trainer.from_argparse_args(args,
                                     callbacks=[ mc, lrm, es],
                                     logger=WandbLogger(name=args.exp_name,
                                                        project='ML_hw7'))
trainer.fit(model, train_loader, val_loader)

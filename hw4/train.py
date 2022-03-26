import csv, os, gc
import json
import random
from pathlib import Path
import sched
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dataset import (InferenceDataset, TrainDataset, get_dataloader,
                     get_dataloader_cv, inference_collate_batch)
from model import Classifier
from utils import get_cosine_schedule_with_warmup, get_amsoftmax_criterion


class Model(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Classifier(args.model_type,
                                pooling=args.pooling,
                                d_model=args.d_model,
                                n_spks=args.speaker_num,
                                dropout=args.dropout,
                                nhead=args.nhead,
                                num_layers=args.num_layers,
                                conv_size=args.conv_size,
                                ffn_dim=args.ffn_dim,
                                amsoftmax=args.amsoftmax)
        if args.amsoftmax:
            self.criterion = get_amsoftmax_criterion(args.am_s, args.am_m)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        self.best_val_acc = -1
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.args.lr,
                                         betas=(0.9, 0.98),
                                         eps=1e-9)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)

        else:
            raise NotImplementedError

        if self.args.scheduler == 'warmup_cosine':
            scheduler = {
                'scheduler':
                get_cosine_schedule_with_warmup(optimizer,
                                                self.args.warmup_steps,
                                                self.args.num_train_steps),
                'interval':
                'step',
                'frequency':
                1
            }
        else:
            scheduler = None    
        return [optimizer], [scheduler]

    def training_step(self, batch, _):
        x, y = batch
        logits = self.model(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log('train/step_loss', loss.item(), prog_bar=True)
        self.log('train/acc', acc.item(), prog_bar=True)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        loss = torch.tensor([o['loss'] for o in outputs]).mean()
        self.log('train/epoch_mean_loss', loss.item(), prog_bar=True)

    def validation_step(self, batch, _):
        x, y = batch
        pred = self.model(x).argmax(-1)
        return {'pred': pred, 'y': y}

    def validation_epoch_end(self, outputs) -> None:
        y = torch.cat([o['y'] for o in outputs]).reshape(-1)
        pred = torch.cat([o['pred'] for o in outputs]).reshape(-1)
        acc = (pred == y).sum() / len(pred)
        self.best_val_acc = max(self.best_val_acc, acc.item())
        self.log_dict({
            'val/acc': acc.item(),
            'val/best_acc': self.best_val_acc
        }, prog_bar=True)


def run_train_cv(args):
    cv = 0

    base_exp_name = args.exp_name
    for train_loader, valid_loader, speaker_num in get_dataloader_cv(
            args.data_dir, args.batch_size, 5, args.segment_len, args.cv):
        cv += 1
        args.speaker_num = speaker_num
        args.exp_name = f'{base_exp_name}_cv-{cv}'
        args.num_train_steps = len(train_loader) * args.max_epochs
        args.warmup_steps = len(train_loader) * args.warmup_epoch
        checkpoint_callback = ModelCheckpoint(
            monitor="val/acc",
            mode='max',
            filename='{epoch}-{other_metric:.2f}' + f'-cv{cv}')
        earlystop = EarlyStopping('val/acc', patience=20, mode='max')
        logger = WandbLogger(project='ml_hw4', name=args.exp_name)
        trainer = pl.Trainer(gpus=1,
                             logger=logger,
                             enable_model_summary=None,
                             reload_dataloaders_every_n_epochs=1,
                             max_epochs=args.max_epochs,
                             callbacks=[
                                 checkpoint_callback, earlystop,
                                 LearningRateMonitor('step')
                             ])
        model = Model(args)
        trainer.fit(model, train_loader, valid_loader)
        logger.experiment.finish()
        del model, trainer
        gc.collect()


def run_train(args):
    train_loader, valid_loader, speaker_num = get_dataloader(
        args.data_dir, args.batch_size, 10, args.segment_len)
    args.speaker_num = speaker_num
    args.num_train_steps = len(train_loader) * args.max_epochs
    args.warmup_steps = len(train_loader) * args.warmup_epoch

    checkpoint_callback = ModelCheckpoint(monitor="val/acc",
                                          mode='max',
                                          every_n_epochs=15)
    earlystop = EarlyStopping('val/acc', patience=20, mode='max')
    logger = WandbLogger(project='ml_hw4', name=args.exp_name)
    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         enable_model_summary=True,
                         reload_dataloaders_every_n_epochs=1,
                         max_epochs=args.max_epochs,
                         gradient_clip_val=0.5,
                         callbacks=[
                             checkpoint_callback, earlystop,
                             LearningRateMonitor('step')
                         ])
    model = Model(args)
    trainer.fit(model, train_loader, valid_loader)
    logger.experiment.finish()
    return checkpoint_callback.best_model_path, model


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def run_test(ckpt, segment_len, data_dir, output_path, device='cuda'):

    model = Model.load_from_checkpoint(ckpt).to(device)
    model.eval()
    dataset = InferenceDataset(data_dir, segment_len)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        collate_fn=inference_collate_batch,
    )
    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    speaker_num = len(mapping["id2speaker"])

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


if __name__ == '__main__':
    from argparse import ArgumentParser
    set_seed(1235)

    parser = ArgumentParser()
    parser.add_argument('--exp_name', default='default')
    parser.add_argument('--data_dir', default='Dataset/')
    parser.add_argument('--segment_len', default=128, type=int)

    parser.add_argument('--model_type', default='conformer')
    parser.add_argument('--pooling', default='attn')
    parser.add_argument('--d_model', type=int, default=120)
    parser.add_argument('--ffn_dim', type=int, default=256)
    parser.add_argument('--conv_size', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--amsoftmax', action='store_true', default=False)
    parser.add_argument('--am_s', type=float, default=15)
    parser.add_argument('--am_m', type=float, default=0.05)
    
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=150)
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--lr', type=float, default=4e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--scheduler', default='warmup_cosine')
    parser.add_argument('--warmup_epoch', default=2, type=int)
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--test_output', default='submit.csv')
    parser.add_argument('--cv', default=0, type=int)
    args = parser.parse_args()
    if args.exp_name == 'default':
        args.exp_name = f'conv_size_{args.conv_size}-d_model_{args.d_model}-ffn_dim_{args.ffn_dim}-lr_{args.lr}-nhead_{args.nhead}-n_layers_{args.num_layers}-ams_{args.amsoftmax}'
    print(args)
    if args.cv < 2:
        best_ckpt_path, _ = run_train(args)
        print(f'best_ckpt_path', best_ckpt_path)
        if args.run_test:
            run_test(best_ckpt_path,
                     args.segment_len,
                     args.data_dir,
                     args.test_output,
                     device='cuda')
    else:
        run_train_cv(args)
    print(args)
    # if args.run_test:
    #     model =

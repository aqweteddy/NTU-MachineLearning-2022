import json
import random
from pathlib import Path
import csv

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
                     inference_collate_batch)
from model import Classifier
from utils import get_cosine_schedule_with_warmup


class Model(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Classifier(args.model_type,
                                pooling=args.pooling,
                                d_model=args.d_model,
                                encoder_dim=args.encoder_dim,
                                n_spks=args.speaker_num,
                                dropout=args.dropout,
                                nhead=args.nhead,
                                num_layers=args.num_layers)
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing)
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
                                          weight_decay=1e-3)

        else:
            raise NotImplementedError

        if self.args.scheduler == 'warmup_cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, self.args.warmup_steps, self.args.num_train_steps)
        else:
            raise NotImplementedError
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
        self.best_val_acc = max(self.best_val_acc, acc)
        self.log('val/acc', acc.item(), prog_bar=True)


def run_train(args):
    train_loader, valid_loader, speaker_num = get_dataloader(
        args.data_dir, args.batch_size, 10)
    args.speaker_num = speaker_num
    args.num_train_steps = len(train_loader)
    args.warmup_steps = len(train_loader) * args.warmup_epoch

    print(args)
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode='max')
    earlystop = EarlyStopping('val/acc', patience=20, mode='max')
    logger = WandbLogger(project='ML_HW4', name=args.exp_name)
    trainer = pl.Trainer(gpus=1,
                         logger=logger,
                         enable_model_summary=None,
                         reload_dataloaders_every_n_epochs=1,
                         max_epochs=args.max_epochs,
                         checkpoint_callback=[
                             checkpoint_callback, earlystop,
                             LearningRateMonitor('step')
                         ])
    model = Model(args)
    trainer.fit(model, train_loader, valid_loader)
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
def run_test(ckpt, data_dir, output_path, device='cuda'):

    model = Model.load_from_checkpoint(ckpt).to(device)
    model.eval()
    dataset = InferenceDataset(data_dir)
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
    set_seed(7777)

    parser = ArgumentParser()
    parser.add_argument('--exp_name', default='default')
    parser.add_argument('--data_dir', default='Dataset/')
    parser.add_argument('--model_type', default='conformer')
    parser.add_argument('--pooling', default='attn')
    parser.add_argument('--d_model', type=int, default=80)
    parser.add_argument('--encoder_dim', type=int, default=144)  # 256, 512
    parser.add_argument('--conv_kernel_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--optimizer', default='adamw')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scheduler', default='warmup_cosine')
    parser.add_argument('--warmup_epoch', default=2, type=int)
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--test_output', default='submit.csv')

    args = parser.parse_args()
    print(args)
    best_ckpt = run_train(args)
    print(f'best_ckpt_path', best_ckpt)
    if args.run_test:
        run_test(best_ckpt, args.data_dir, args.test_output)

    # if args.run_test:
    #     model =
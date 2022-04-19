import pytorch_lightning as pl
import torch
from transformers import get_cosine_schedule_with_warmup, BertForQuestionAnswering, get_constant_schedule_with_warmup


class BaseTrainer(pl.LightningModule):

    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(),
                                self.hparams.lr,
                                weight_decay=1e-3)
        if self.hparams.scheduler in [None, 'none']:
            return opt
        elif self.hparams.scheduler == 'cosine_warmup':
            sch = {
                'scheduler':
                get_cosine_schedule_with_warmup(opt, self.hparams.warmup_steps,
                                                self.hparams.num_train_steps),
                'interval':
                'step'
            }
        elif self.hparams.scheduler == 'constant_warmup':
            sch = {
                'scheduler':
                get_constant_schedule_with_warmup(
                    opt, self.hparams.warmup_steps),
                'interval':
                'step'
            }
        else:
            raise NotImplementedError
        return [opt], [sch]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("train")
        parser.add_argument('--pretrained',
                            default='ckiplab/bert-base-chinese',
                            type=str)
        parser.add_argument('--scheduler', default=None, type=str)
        parser.add_argument('--lr', default=2e-5, type=float)
        parser.add_argument('--warmup_epochs', default=1.5, type=float)
        return parent_parser



class QATrainer(BaseTrainer):

    def __init__(self, args) -> None:
        super().__init__(args)
        self.model = BertForQuestionAnswering.from_pretrained(
            self.hparams.pretrained)
        self.best_em = 0.

    @torch.no_grad()
    def forward(self, input_ids, token_ids, attn_mask):
        result = self.model(input_ids, attn_mask, token_ids)
        return result.start_logits, result.end_logits

    def training_step(self, batch, batch_idx):
        input_ids, token_ids, attn_mask, start_pos, end_pos = batch
        loss = self.model(input_ids,
                          attn_mask,
                          token_ids,
                          start_positions=start_pos,
                          end_positions=end_pos).loss
        self.log('train/loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, token_ids, attn_mask, start_pos, end_pos = batch
        pred_start_logits, pred_end_logits = self(input_ids, token_ids, attn_mask)
        pred_start_pos = pred_start_logits.argmax(-1)
        pred_end_pos = pred_end_logits.argmax(-1)
        mask = (start_pos == pred_start_pos) & (end_pos == pred_end_pos)
        return {'EM': mask}

    def validation_epoch_end(self, outputs) -> None:
        em = torch.cat([o['EM'] for o in outputs], dim=-1).reshape(-1)
        em = em.float().mean()
        self.best_em = max(self.best_em, em.item())
        self.log('val/EM', em, prog_bar=True)
        self.log('val/best_EM', self.best_em, prog_bar=True)

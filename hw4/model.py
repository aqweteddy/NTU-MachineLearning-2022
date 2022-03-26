from torch import dropout, nn
import torch
from torch.nn import functional as F
# from conformer.encoder import ConformerEncoder
from conformer_torchautdio import Conformer


class Pooling(nn.Module):

    def __init__(self, method='attn') -> None:
        super().__init__()
        if method == 'attn':
            self.proj = nn.LazyLinear(1, bias=False)
        self.method = method

    def forward(self, x: torch.tensor):
        """
        x: [B, S, E]
        """
        if self.method == 'attn':
            weight = torch.softmax(self.proj(x).squeeze(-1), dim=-1)  # [B, S]
            return x.transpose(1, 2).bmm(weight.unsqueeze(-1)).squeeze(-1)
        else:
            return x.mean(1)


class Prediction(nn.Module):

    def __init__(self, inp_size, out_size, amsoftmax=False):
        super().__init__()
        self.fc = nn.Linear(inp_size, out_size, bias=False)
        self.amsoftmax = amsoftmax

    def forward(self, x):
        if self.amsoftmax:
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)
        logits = self.fc(x)
        return logits


class Classifier(nn.Module):

    def __init__(self,
                 encoder_type,
                 d_model,
                 pooling='attn',
                 n_spks=600,
                 dropout=0.2,
                 ffn_dim=256,
                 conv_size=32,
                 nhead=8,
                 num_layers=4,
                 amsoftmax=False):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)
        self.dropout = dropout
        if encoder_type == 'conformer':
            self.encoder = Conformer(d_model, nhead, ffn_dim, num_layers,
                                     conv_size, dropout)
        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                       nhead=nhead,
                                                       dim_feedforward=ffn_dim,
                                                       dropout=dropout)
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        self.encoder_type = encoder_type
        self.pooling = Pooling(pooling)
        self.pred = Prediction(d_model, n_spks, amsoftmax)

    def forward(self, mels):
        """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
        out = self.prenet(mels)
        F.dropout(out, self.dropout, inplace=True)
        if self.encoder_type == 'conformer':
            out, _ = self.encoder(
                out,
                torch.tensor([mels.shape[1]] * mels.shape[0]).to(out.device))
        elif self.encoder_type == 'transformer':
            out = self.encoder(out)

        stats = self.pooling(out)
        out = self.pred(stats)
        return out

from torch import nn
import torch
from torch.nn import functional as F
from conformer.encoder import ConformerEncoder



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
            weight = torch.softmax(self.proj(x).squeeze(-1), dim=-1) # [B, S]
            return x.transpose(1, 2).bmm(weight.unsqueeze(-1)).squeeze(-1)
        else:
            return x.mean(1)


class Classifier(nn.Module):

    def __init__(self,
                 encoder_type,
                 d_model,
                 pooling='attn',
                 n_spks=600,
                 dropout=0.2,
                 encoder_dim=512,
                 nhead=8,
                 num_layers=4):
        super().__init__()
        self.prenet = nn.Linear(40, d_model)
        if encoder_type == 'conformer':
            self.encoder = ConformerEncoder(d_model,
                                            encoder_dim=encoder_dim,
                                            num_layers=num_layers,
                                            num_attention_heads=nhead,
                                            input_dropout_p=dropout,
                                            attention_dropout_p=dropout,
                                            conv_dropout_p=dropout)

        elif encoder_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=encoder_dim,
                dropout=dropout
            )
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers,
                                                 )
        self.encoder_type = encoder_type
        self.pooling = Pooling(pooling)
        self.pred_layer = nn.Sequential(nn.Linear(encoder_dim, n_spks), )

    def forward(self, mels):
        """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
        out = self.prenet(mels)
        if self.encoder_type == 'conformer':
            out, _ = self.encoder(out, mels.shape[1])
        elif self.encoder_type == 'transformer':
            out = self.encoder(out)
        
        stats = self.pooling(out)
        out = self.pred_layer(stats)
        return out

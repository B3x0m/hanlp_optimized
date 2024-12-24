import torch
from torch import nn
from typing import List
from hanlp.components.parsers.constituency.treecrf import CRFConstituency
from hanlp.components.parsers.alg import cky
from hanlp.components.parsers.biaffine.biaffine import Biaffine
from hanlp.components.parsers.biaffine.mlp import MLP


class CRFConstituencyDecoder(nn.Module):
    def __init__(self,
                 n_labels,
                 n_hidden=400,
                 n_mlp_span=500,
                 n_mlp_label=100,
                 mlp_dropout=.33,
                 **kwargs
                 ):
        super().__init__()

        self.mlp_span_l = MLP(n_in=n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_span_r = MLP(n_in=n_hidden, n_out=n_mlp_span, dropout=mlp_dropout)
        self.mlp_label_l = MLP(n_in=n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)
        self.mlp_label_r = MLP(n_in=n_hidden, n_out=n_mlp_label, dropout=mlp_dropout)

        self.span_attn = Biaffine(n_in=n_mlp_span, bias_x=True, bias_y=False)
        self.label_attn = Biaffine(n_in=n_mlp_label, n_out=n_labels, bias_x=True, bias_y=True)
        self.crf = CRFConstituency()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        x_f, x_b = x.chunk(2, -1)
        x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        span_l = self.mlp_span_l(x)
        span_r = self.mlp_span_r(x)
        label_l = self.mlp_label_l(x)
        label_r = self.mlp_label_r(x)

        s_span = self.span_attn(span_l, span_r)
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label

    def loss(self, s_span, s_label, charts, mask, mbr=True):
        span_mask = charts.ge(0) & mask
        span_loss, span_probs = self.crf(s_span, mask, span_mask, mbr)
        label_loss = self.criterion(s_label[span_mask], charts[span_mask])
        loss = span_loss + label_loss

        return loss, span_probs

    def decode(self, s_span, s_label, mask):
        span_preds = cky(s_span, mask)
        label_preds = s_label.argmax(-1).tolist()
        return [[(i, j, labels[i][j]) for i, j in spans] for spans, labels in zip(span_preds, label_preds)]


class CRFConstituencyModel(nn.Module):

    def __init__(self, encoder, decoder: CRFConstituencyDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch):
        x = self.encoder(batch)
        return self.decoder(x)
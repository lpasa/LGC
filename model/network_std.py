from torch import nn
import torch
import dgl


class GCNetwork(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_classes,
                 dropout,
                 k,
                 convLayer,
                 out_fun = nn.Softmax(dim=1),
                 device=None,
                 norm=None,
                 bias=False):
        super(GCNetwork, self).__init__()
        self.g=g


        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device=device
        self.layers = nn.ModuleList()
        self.out_fun=out_fun
        self.dropout = nn.Dropout(p=dropout)
        self.layers.append(convLayer(in_feats, n_classes, k, cached=False, bias=bias, norm=norm))#cached=True

    def forward(self,features):
        h = features
        for layer in self.layers:
            h=self.dropout(h)
            h = layer(self.g,h)
        return self.out_fun(h),h

import numbers

import torch
import torch.nn as nn


def select_norm(norm, dim, eps=1e-5, group=1):
    if norm not in ["cLN", "LN", "gLN", "GN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "LN":
        return nn.LayerNorm(dim, eps, elementwise_affine=True)
    elif norm == "GN":
        return nn.GroupNorm(group, dim, eps)
    elif norm == "BN":
        return nn.BatchNorm1d(dim, eps)
    else:
        return GlobalChannelLayerNorm(dim, eps, elementwise_affine=True)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation (FiLM) layer"""

    def __init__(self, feat_size, embed_size, num_film_layers=1, layer_norm=False):
        super(FiLM, self).__init__()
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.num_film_layers = num_film_layers
        self.layer_norm = nn.LayerNorm(embed_size) if layer_norm else None
        gamma_fcs, beta_fcs = [], []
        for i in range(num_film_layers):
            if i == 0:
                gamma_fcs.append(nn.Linear(embed_size, feat_size))
                beta_fcs.append(nn.Linear(embed_size, feat_size))
            else:
                gamma_fcs.append(nn.Linear(feat_size, feat_size))
                beta_fcs.append(nn.Linear(feat_size, feat_size))
        self.gamma_fcs = nn.ModuleList(gamma_fcs)
        self.beta_fcs = nn.ModuleList(beta_fcs)
        self.init_weights()

    def init_weights(self):
        for i in range(self.num_film_layers):
            nn.init.zeros_(self.gamma_fcs[i].weight)
            nn.init.zeros_(self.gamma_fcs[i].bias)
            nn.init.zeros_(self.beta_fcs[i].weight)
            nn.init.zeros_(self.beta_fcs[i].bias)

    def forward(self, embed, x):
        gamma, beta = None, None
        for i in range(len(self.gamma_fcs)):
            if i == 0:
                gamma = self.gamma_fcs[i](embed)
                beta = self.beta_fcs[i](embed)
            else:
                gamma = self.gamma_fcs[i](gamma)
                beta = self.beta_fcs[i](beta)
        if len(gamma.shape) < len(x.shape):
            gamma = gamma.unsqueeze(-1).expand_as(x)
            beta = beta.unsqueeze(-1).expand_as(x)
        else:
            gamma = gamma.expand_as(x)
            beta = beta.expand_as(x)
        x = (1 + gamma) * x + beta
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = torch.transpose(x, 1, -1)
        x = super().forward(x)
        x = torch.transpose(x, 1, -1)
        return x


class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(self.__name__))
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        if self.elementwise_affine:
            x = (self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias)
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x
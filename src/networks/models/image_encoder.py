""" Image encoder based on PVSE implementation.
Reference code:
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import sys

import torch.nn as nn
from torchvision import models

sys.path.append("../")
sys.path.append("../../")
from src.networks.models.pie_model import PIENet
from src.networks.models.uncertainty_module import UncertaintyModuleImage
from src.utils.tensor_utils import l2_normalize, sample_gaussian_tensors


class EncoderImage(nn.Module):
    def __init__(self, config, mlp_local):
        super(EncoderImage, self).__init__()

        embed_dim = config.embed_dim

        # Backbone CNN
        self.cnn = getattr(models, config.cnn_type)(pretrained=True)
        cnn_dim = self.cnn_dim = self.cnn.fc.in_features

        self.avgpool = self.cnn.avgpool
        self.cnn.avgpool = nn.Sequential()

        self.fc = nn.Linear(cnn_dim, embed_dim)

        self.cnn.fc = nn.Sequential()

        self.pie_net = PIENet(1, cnn_dim, embed_dim, cnn_dim // 2)

        for idx, param in enumerate(self.cnn.parameters()):
            param.requires_grad = True

        self.n_samples_inference = config.get('n_samples_inference', 0)

        self.mlp_local = mlp_local
        if self.mlp_local:
            self.head_proj = nn.Sequential(
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512)
            )

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, images):
        out_7x7 = self.cnn(images).view(-1, self.cnn_dim, 7, 7)
        pooled = self.avgpool(out_7x7).view(-1, self.cnn_dim)
        out = self.fc(pooled)

        output = {}
        out_7x7 = out_7x7.view(-1, self.cnn_dim, 7 * 7)

        out, attn, residual = self.pie_net(out, out_7x7.transpose(1, 2))

        if self.mlp_local:
            out = self.head_proj(out)

        out = l2_normalize(out)

        output['embedding'] = out

        return output

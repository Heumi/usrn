import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args):
    return NORMAL(args)


class NORMAL(nn.Module):
    def __init__(self, config, conv=common.default_conv):
        super(NORMAL, self).__init__()
        self.scale = config.scale
        in_channels = config.in_channels
        n_feats = config.n_feats
        kernel_size = 3

        # define head module
        head = [common.default_conv(in_channels, n_feats, kernel_size)]

        # define body module
        body = [common.ResBlock(conv, n_feats, kernel_size)
                for _ in range(3)]
        body.append(conv(n_feats, n_feats, kernel_size))

        # define upsample module
        upsample = [common.Upsampler(conv, self.scale, n_feats, act=False),
                    conv(n_feats, in_channels, kernel_size)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.upsample = nn.Sequential(*upsample)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/self.scale)
        x_head = self.head(x)
        x_body = self.body(x_head)
        x_mean = self.upsample(x_body)

        results = {'mean': x_mean}
        return results

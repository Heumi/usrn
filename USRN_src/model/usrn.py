import torch.nn as nn
import torch.nn.functional as F

from model import common


def make_model(args):
    return USRN(args)


class USRN(nn.Module):
    def __init__(self, config, conv=common.default_conv):
        super(USRN, self).__init__()
        self.scale = config.scale
        in_channels = config.in_channels
        n_feats = config.n_feats
        kernel_size = 3

        # define head module
        head = [common.default_conv(in_channels, n_feats, kernel_size)]

        # define body module
        body_mid = [common.ResBlock(conv, n_feats, kernel_size)
                    for _ in range(2)]

        body_mean = [common.ResBlock(conv, n_feats, kernel_size)
                     for _ in range(2)]
        body_mean.append(conv(n_feats, n_feats, kernel_size))

        body_var = [common.ResBlock(conv, n_feats, kernel_size)
                    for _ in range(2)]
        body_var.append(conv(n_feats, n_feats, kernel_size))

        # define upsample module
        upsample_mean = [common.Upsampler(conv, self.scale, n_feats, act=False),
                    conv(n_feats, in_channels, kernel_size)]
        upsample_var = [common.Upsampler(conv, self.scale, n_feats, act=False),
                    conv(n_feats, in_channels, kernel_size)]

        self.head = nn.Sequential(*head)
        self.body_mid = nn.Sequential(*body_mid)
        self.body_mean = nn.Sequential(*body_mean)
        self.upsample_mean = nn.Sequential(*upsample_mean)
        self.body_var = nn.Sequential(*body_var)
        self.upsample_var = nn.Sequential(*upsample_var)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=1/self.scale)
        x_head = self.head(x)
        x_body_mid = self.body_mid(x_head)

        x_body_mean = self.body_mean(x_body_mid)
        x_mean = self.upsample_mean(x_body_mean)

        x_body_var = self.body_var(x_body_mid)
        x_var = self.upsample_var(x_body_var)

        results = {'mean': x_mean, 'var': x_var}
        return results

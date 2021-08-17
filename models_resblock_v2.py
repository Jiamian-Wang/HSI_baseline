import torch
import torch.nn as nn
from architecture.ResidualFeat import Res2Net
from architecture.netunit import *
import common


_NORM_BONE = False

class SSI_RES_UNET(nn.Module):

    def __init__(self, in_ch=28, out_ch=28, conv=common.default_conv):
        super(SSI_RES_UNET, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3
        scale = 1
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(in_ch, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale= 1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [conv(n_feats, out_ch, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

        return x





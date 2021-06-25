import torch
import torch.nn as nn
from architecture.ResidualFeat import Res2Net
from architecture.netunit import *
import common


_NORM_BONE = False

class my_model(nn.Module):

    def __init__(self, in_ch=28, out_ch=28, conv=common.default_conv):
        super(my_model, self).__init__()

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




# class Encoder_Triblock(nn.Module):
#     def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
#         super(Encoder_Triblock, self).__init__()
#
#         self.layer1 = conv_block(inChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
#         if flag_res:
#             self.layer2 = Res2Net(outChannel,int(outChannel/4))
#         else:
#             self.layer2 = conv_block(outChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
#
#         self.pool = nn.MaxPool2d(nPool) if flag_Pool else None
#     def forward(self,x):
#         feat = self.layer1(x)
#         feat = self.layer2(feat)
#
#         feat_pool = self.pool(feat) if self.pool is not None else feat
#         return feat_pool,feat
#
# class Decoder_Triblock(nn.Module):
#     def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
#         super(Decoder_Triblock, self).__init__()
#
#         self.layer1 = nn.Sequential(
#             nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )
#         if flag_res:
#             self.layer2 = Res2Net(int(outChannel*2),int(outChannel/2))
#         else:
#             self.layer2 = conv_block(outChannel*2,outChannel*2,nKernal,flag_norm=_NORM_BONE)
#         self.layer3 = conv_block(outChannel*2,outChannel,nKernal,flag_norm=_NORM_BONE)
#
#     def forward(self,feat_dec,feat_enc):
#         feat_dec = self.layer1(feat_dec)
#         diffY = feat_enc.size()[2] - feat_dec.size()[2]
#         diffX = feat_enc.size()[3] - feat_dec.size()[3]
#         if diffY != 0 or diffX != 0:
#             print('Padding for size mismatch ( Enc:', feat_enc.size(), 'Dec:', feat_dec.size(),')')
#             feat_dec = F.pad(feat_dec, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
#         feat = torch.cat([feat_dec,feat_enc],dim=1)
#         feat = self.layer2(feat)
#         feat = self.layer3(feat)
#         return feat
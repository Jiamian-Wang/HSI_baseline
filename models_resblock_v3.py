import torch
import torch.nn as nn
from architecture.TSA_Module import TSA_Transform
from architecture.ResidualFeat import Res2Net
from architecture.netunit import *
import common                                                       # add

import pdb

_NORM_BONE = False

class EDSR(nn.Module):                                              # add new EDSR model
    def __init__(self, in_ch=28, out_ch=28, conv=common.default_conv):   # rm, args
        super(EDSR, self).__init__()

        n_resblocks = 16                                            # change, args.n_resblocks
        n_feats = 64                                                # change, args.n_feats
        kernel_size = 3
        scale = 2                                                    # change, args.scale[0]
        act = nn.ReLU(True)
        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale) # rm
        # if url_name in url:                                        # rm
        #     self.url = url[url_name]                               # rm
        # else:                                                      # rm
        #     self.url = None                                        # rm
        # self.sub_mean = common.MeanShift(args.rgb_range)           # rm
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)   # rm

        # define head module
        m_head = [conv(in_ch, n_feats, kernel_size),
                  nn.Conv2d(n_feats, n_feats, 3, stride=2, padding=1)]               # change, args.n_colors

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale= 1
            ) for _ in range(n_resblocks)                             # change, args.res_scale
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [common.Upsampler(conv, scale, n_feats, act=False),
                  conv(n_feats, out_ch, kernel_size)]                 # change, args.n_colors

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

# class TSA_Net(nn.Module):            # rm TSA_Net
#
#     def __init__(self,in_ch=28, out_ch=28):
#         super(TSA_Net, self).__init__()
#
#         self.tconv_down1 = Encoder_Triblock(in_ch, 64, False)
#         self.tconv_down2 = Encoder_Triblock(64, 128, False)
#         self.tconv_down3 = Encoder_Triblock(128, 256)
#         self.tconv_down4 = Encoder_Triblock(256, 512)
#
#         self.bottom1 = conv_block(512,1024)
#         self.bottom2 = conv_block(1024,1024)
#
#         self.tconv_up4 = Decoder_Triblock(1024, 512)
#         self.tconv_up3 = Decoder_Triblock(512, 256)
#         # self.transform3 = TSA_Transform((64,64),256,256,8,(64,80),[0,0]) # rm
#         self.transform3_1 = ResBlock(256, 3) # add
#         self.transform3_2 = ResBlock(256, 3)  # add
#         self.transform3_3 = ResBlock(256, 3)  # add
#         self.transform3_4 = ResBlock(256, 3)  # add
#         self.tconv_up2 = Decoder_Triblock(256, 128)
#         # self.transform2 = TSA_Transform((128,128),128,128,8,(64,40),[1,0]) # rm
#         self.transform2_1 = ResBlock(128, 3) # add
#         self.transform2_2 = ResBlock(128, 3)  # add
#         self.transform2_3 = ResBlock(128, 3)  # add
#         self.transform2_4 = ResBlock(128, 3)  # add
#         self.tconv_up1 = Decoder_Triblock(128, 64)
#         # self.transform1 = TSA_Transform((256,256),64,28,8,(48,30),[1,1],True) # rm
#         self.transform1_1 = ResBlock(64, 3) # add
#         self.transform1_2 = ResBlock(64, 3)  # add
#         self.transform1_3 = ResBlock(64, 3)  # add
#         self.transform1_4 = nn.Conv2d(64, 28, 3, 1, 1)  # add
#
#         self.conv_last = nn.Conv2d(out_ch, out_ch, 1)
#         self.afn_last = nn.Sigmoid()
#
#
#     def forward(self, x):
#         enc1,enc1_pre = self.tconv_down1(x)
#         enc2,enc2_pre = self.tconv_down2(enc1)
#         enc3,enc3_pre = self.tconv_down3(enc2)
#         enc4,enc4_pre = self.tconv_down4(enc3)
#         #enc5,enc5_pre = self.tconv_down5(enc4)
#
#         bottom = self.bottom1(enc4)
#         bottom = self.bottom2(bottom)
#
#         #dec5 = self.tconv_up5(bottom,enc5_pre)
#         dec4 = self.tconv_up4(bottom,enc4_pre)
#         dec3 = self.tconv_up3(dec4,enc3_pre)
#         dec3 = self.transform3_1(dec3) # change
#         dec3 = self.transform3_2(dec3) # add
#         dec3 = self.transform3_3(dec3) # add
#         dec3 = self.transform3_4(dec3)  # add
#         dec2 = self.tconv_up2(dec3,enc2_pre)
#         dec2 = self.transform2_1(dec2) # change
#         dec2 = self.transform2_2(dec2) # add
#         dec2 = self.transform2_3(dec2) # add
#         dec2 = self.transform2_4(dec2)  # add
#         dec1 = self.tconv_up1(dec2,enc1_pre)
#         dec1 = self.transform1_1(dec1) # change
#         dec1 = self.transform1_2(dec1)  # add
#         dec1 = self.transform1_3(dec1)  # add
#         dec1 = self.transform1_4(dec1)  # add
#
#         dec1 = self.conv_last(dec1)
#         output = self.afn_last(dec1)
#
#         return output

# add, to use res_block
# class ResBlock(nn.Module):
#     def __init__(
#         self, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.ReLU(True), res_scale=1): # change
#
#         super(ResBlock, self).__init__()
#         m = []
#
#         for i in range(2):
#             m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), bias=bias)) # change
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if i == 0:
#                 m.append(act)
#         self.body = nn.Sequential(*m)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x).mul(self.res_scale)
#         res += x
#         return res



class Encoder_Triblock(nn.Module):
    def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
        super(Encoder_Triblock, self).__init__()

        self.layer1 = conv_block(inChannel,outChannel,nKernal,flag_norm=_NORM_BONE)
        if flag_res:
            self.layer2 = Res2Net(outChannel,int(outChannel/4))
        else:
            self.layer2 = conv_block(outChannel,outChannel,nKernal,flag_norm=_NORM_BONE)

        self.pool = nn.MaxPool2d(nPool) if flag_Pool else None
    def forward(self,x):
        feat = self.layer1(x)
        feat = self.layer2(feat)

        feat_pool = self.pool(feat) if self.pool is not None else feat            
        return feat_pool,feat

class Decoder_Triblock(nn.Module):
    def __init__(self,inChannel,outChannel,flag_res=True,nKernal=3,nPool=2,flag_Pool=True):
        super(Decoder_Triblock, self).__init__()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(inChannel, outChannel, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        if flag_res:
            self.layer2 = Res2Net(int(outChannel*2),int(outChannel/2))
        else:
            self.layer2 = conv_block(outChannel*2,outChannel*2,nKernal,flag_norm=_NORM_BONE)
        self.layer3 = conv_block(outChannel*2,outChannel,nKernal,flag_norm=_NORM_BONE)

    def forward(self,feat_dec,feat_enc):
        feat_dec = self.layer1(feat_dec)
        diffY = feat_enc.size()[2] - feat_dec.size()[2]
        diffX = feat_enc.size()[3] - feat_dec.size()[3]
        if diffY != 0 or diffX != 0:
            print('Padding for size mismatch ( Enc:', feat_enc.size(), 'Dec:', feat_dec.size(),')')
            feat_dec = F.pad(feat_dec, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        feat = torch.cat([feat_dec,feat_enc],dim=1)
        feat = self.layer2(feat)
        feat = self.layer3(feat)
        return feat
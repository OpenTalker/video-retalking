'''
@Created by chaofengc (chaofenghust@gmail.com)

@Modified by yangxy (yangtao9009@gmail.com)
'''

from face_parse.blocks import *
import torch
from torch import nn
import numpy as np

def define_P(in_size=512, out_size=512, min_feat_size=32, relu_type='LeakyReLU', isTrain=False, weight_path=None):
    net = ParseNet(in_size, out_size, min_feat_size, 64, 19, norm_type='bn', relu_type=relu_type, ch_range=[32, 256])
    if not isTrain:
        net.eval()  
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path))
    return net


class ParseNet(nn.Module):
    def __init__(self,
                in_size=128,
                out_size=128,
                min_feat_size=32,
                base_ch=64,
                parsing_ch=19,
                res_depth=10,
                relu_type='prelu',
                norm_type='bn',
                ch_range=[32, 512],
                ):
        super().__init__()
        self.res_depth = res_depth
        act_args = {'norm_type': norm_type, 'relu_type': relu_type}
        min_ch, max_ch = ch_range

        ch_clip = lambda x: max(min_ch, min(x, max_ch))
        min_feat_size = min(in_size, min_feat_size)

        down_steps = int(np.log2(in_size//min_feat_size))
        up_steps = int(np.log2(out_size//min_feat_size))

        # =============== define encoder-body-decoder ==================== 
        self.encoder = []
        self.encoder.append(ConvLayer(3, base_ch, 3, 1))
        head_ch = base_ch
        for i in range(down_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', **act_args))
            head_ch = head_ch * 2

        self.body = []
        for i in range(res_depth):
            self.body.append(ResidualBlock(ch_clip(head_ch), ch_clip(head_ch), **act_args))

        self.decoder = []
        for i in range(up_steps):
            cin, cout = ch_clip(head_ch), ch_clip(head_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', **act_args))
            head_ch = head_ch // 2

        self.encoder = nn.Sequential(*self.encoder)
        self.body = nn.Sequential(*self.body)
        self.decoder = nn.Sequential(*self.decoder)
        self.out_img_conv = ConvLayer(ch_clip(head_ch), 3)
        self.out_mask_conv = ConvLayer(ch_clip(head_ch), parsing_ch)

    def forward(self, x):
        feat = self.encoder(x)
        x = feat + self.body(feat)
        x = self.decoder(x)
        out_img = self.out_img_conv(x) 
        out_mask = self.out_mask_conv(x)
        return out_mask, out_img



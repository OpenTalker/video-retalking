import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_blocks import ResBlock, StyleConv, ToRGB


class ENet(nn.Module):
    def __init__(
        self, 
        num_style_feat=512,
        lnet=None,
        concat=False
        ):  
        super(ENet, self).__init__()

        self.low_res = lnet
        for param in self.low_res.parameters():
            param.requires_grad = False

        channel_multiplier, narrow = 2, 1
        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }

        self.log_size = 8
        first_out_size = 128
        self.conv_body_first = nn.Conv2d(3, channels[f'{first_out_size}'], 1) # 256 -> 128

        # downsample
        in_channels = channels[f'{first_out_size}']
        self.conv_body_down = nn.ModuleList()
        for i in range(8, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            self.conv_body_down.append(ResBlock(in_channels, out_channels, mode='down'))
            in_channels = out_channels

        self.num_style_feat = num_style_feat
        linear_out_channel = num_style_feat
        self.final_linear = nn.Linear(channels['4'] * 4 * 4, linear_out_channel)
        self.final_conv = nn.Conv2d(in_channels, channels['4'], 3, 1, 1)

        self.style_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        
        self.concat = concat
        if concat:
            in_channels = 3 + 32 # channels['64']
        else:
            in_channels = 3

        for i in range(7, 9):  # 128, 256
            out_channels = channels[f'{2**i}'] # 
            self.style_convs.append(
                StyleConv(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode='upsample'))
            self.style_convs.append(
                StyleConv(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    num_style_feat=num_style_feat,
                    demodulate=True,
                    sample_mode=None))
            self.to_rgbs.append(ToRGB(out_channels, num_style_feat, upsample=True))
            in_channels = out_channels

    def forward(self, audio_sequences, face_sequences, gt_sequences):
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())
        inp, ref = torch.split(face_sequences,3,dim=1)

        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            inp = torch.cat([inp[:, :, i] for i in range(inp.size(2))], dim=0)
            ref = torch.cat([ref[:, :, i] for i in range(ref.size(2))], dim=0)
            gt_sequences = torch.cat([gt_sequences[:, :, i] for i in range(gt_sequences.size(2))], dim=0)
        
        # get the global style
        feat = F.leaky_relu_(self.conv_body_first(F.interpolate(ref, size=(256,256), mode='bilinear')), negative_slope=0.2)
        for i in range(self.log_size - 2):
            feat = self.conv_body_down[i](feat)
        feat = F.leaky_relu_(self.final_conv(feat), negative_slope=0.2)

        # style code
        style_code = self.final_linear(feat.reshape(feat.size(0), -1))
        style_code = style_code.reshape(style_code.size(0), -1, self.num_style_feat)
        
        LNet_input = torch.cat([inp, gt_sequences], dim=1)
        LNet_input = F.interpolate(LNet_input, size=(96,96), mode='bilinear')
        
        if self.concat:
            low_res_img, low_res_feat = self.low_res(audio_sequences, LNet_input)
            low_res_img.detach()
            low_res_feat.detach()
            out = torch.cat([low_res_img, low_res_feat], dim=1) 

        else:
            low_res_img = self.low_res(audio_sequences, LNet_input)
            low_res_img.detach()
            # 96 x 96
            out = low_res_img 
        
        p2d = (2,2,2,2)
        out = F.pad(out, p2d, "reflect", 0)
        skip = out

        for conv1, conv2, to_rgb in zip(self.style_convs[::2], self.style_convs[1::2], self.to_rgbs):
            out = conv1(out, style_code)  # 96, 192, 384
            out = conv2(out, style_code)
            skip = to_rgb(out, style_code, skip)
        _outputs = skip

        # remove padding
        _outputs = _outputs[:,:,8:-8,8:-8]

        if input_dim_size > 4:
            _outputs = torch.split(_outputs, B, dim=0)
            outputs = torch.stack(_outputs, dim=2)
            low_res_img = F.interpolate(low_res_img, outputs.size()[3:])
            low_res_img = torch.split(low_res_img, B, dim=0) 
            low_res_img = torch.stack(low_res_img, dim=2)
        else:
            outputs = _outputs
        return outputs, low_res_img
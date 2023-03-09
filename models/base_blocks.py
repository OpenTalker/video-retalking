import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from models.ffc import FFC
from basicsr.archs.arch_util import default_init_weights


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='down'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if mode == 'down':
            self.scale_factor = 0.5
        elif mode == 'up':
            self.scale_factor = 2

    def forward(self, x):
        out = F.leaky_relu_(self.conv1(x), negative_slope=0.2)
        # upsample/downsample
        out = F.interpolate(out, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        out = F.leaky_relu_(self.conv2(out), negative_slope=0.2)
        # skip
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        skip = self.skip(x)
        out = out + skip
        return out


class LayerNorm2d(nn.Module):
    def __init__(self, n_out, affine=True):
        super(LayerNorm2d, self).__init__()
        self.n_out = n_out
        self.affine = affine

        if self.affine:
          self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
          self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))

    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
          return F.layer_norm(x, normalized_shape, \
              self.weight.expand(normalized_shape), 
              self.bias.expand(normalized_shape))    
        else:
          return F.layer_norm(x, normalized_shape)  


def spectral_norm(module, use_spect=True):
    if use_spect:
        return SpectralNorm(module)
    else:
        return module


class FirstBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FirstBlock2d, self).__init__()
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding': 3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out 


class DownBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(DownBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        pool = nn.AvgPool2d(kernel_size=(2, 2))

        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity, pool)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity, pool)

    def forward(self, x):
        out = self.model(x)
        return out 


class UpBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(UpBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(output_nc), nonlinearity)

    def forward(self, x):
        out = self.model(F.interpolate(x, scale_factor=2))
        return out


class ADAIN(nn.Module):
    def __init__(self, norm_nc, feature_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        nhidden = 128
        use_bias=True

        self.mlp_shared = nn.Sequential(
            nn.Linear(feature_nc, nhidden, bias=use_bias),            
            nn.ReLU()
        )
        self.mlp_gamma = nn.Linear(nhidden, norm_nc, bias=use_bias)    
        self.mlp_beta = nn.Linear(nhidden, norm_nc, bias=use_bias)    

    def forward(self, x, feature):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on feature
        feature = feature.view(feature.size(0), -1)
        actv = self.mlp_shared(feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.view(*gamma.size()[:2], 1,1)
        beta = beta.view(*beta.size()[:2], 1,1)
        out = normalized * (1 + gamma) + beta
        return out


class FineADAINResBlock2d(nn.Module):
    """
    Define an Residual block for different types
    """
    def __init__(self, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlock2d, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.conv1 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.conv2 = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        self.norm1 = ADAIN(input_nc, feature_nc)
        self.norm2 = ADAIN(input_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        dx = self.actvn(self.norm1(self.conv1(x), z))
        dx = self.norm2(self.conv2(x), z)
        out = dx + x
        return out  


class FineADAINResBlocks(nn.Module):
    def __init__(self, num_block, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINResBlocks, self).__init__()                                
        self.num_block = num_block
        for i in range(num_block):
            model = FineADAINResBlock2d(input_nc, feature_nc, norm_layer, nonlinearity, use_spect)
            setattr(self, 'res'+str(i), model)

    def forward(self, x, z):
        for i in range(self.num_block):
            model = getattr(self, 'res'+str(i))
            x = model(x, z)
        return x   


class ADAINEncoderBlock(nn.Module):       
    def __init__(self, input_nc, output_nc, feature_nc, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoderBlock, self).__init__()
        kwargs_down = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_fine = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv_0 = spectral_norm(nn.Conv2d(input_nc,  output_nc, **kwargs_down), use_spect)
        self.conv_1 = spectral_norm(nn.Conv2d(output_nc, output_nc, **kwargs_fine), use_spect)


        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(output_nc, feature_nc)
        self.actvn = nonlinearity

    def forward(self, x, z):
        x = self.conv_0(self.actvn(self.norm_0(x, z)))
        x = self.conv_1(self.actvn(self.norm_1(x, z)))
        return x


class ADAINDecoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, hidden_nc, feature_nc, use_transpose=True, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINDecoderBlock, self).__init__()        
        # Attributes
        self.actvn = nonlinearity
        hidden_nc = min(input_nc, output_nc) if hidden_nc is None else hidden_nc

        kwargs_fine = {'kernel_size':3, 'stride':1, 'padding':1}
        if use_transpose:
            kwargs_up = {'kernel_size':3, 'stride':2, 'padding':1, 'output_padding':1}
        else:
            kwargs_up = {'kernel_size':3, 'stride':1, 'padding':1}

        # create conv layers
        self.conv_0 = spectral_norm(nn.Conv2d(input_nc, hidden_nc, **kwargs_fine), use_spect)
        if use_transpose:
            self.conv_1 = spectral_norm(nn.ConvTranspose2d(hidden_nc, output_nc, **kwargs_up), use_spect)
            self.conv_s = spectral_norm(nn.ConvTranspose2d(input_nc, output_nc, **kwargs_up), use_spect)
        else:
            self.conv_1 = nn.Sequential(spectral_norm(nn.Conv2d(hidden_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
            self.conv_s = nn.Sequential(spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs_up), use_spect),
                                        nn.Upsample(scale_factor=2))
        # define normalization layers
        self.norm_0 = ADAIN(input_nc, feature_nc)
        self.norm_1 = ADAIN(hidden_nc, feature_nc)
        self.norm_s = ADAIN(input_nc, feature_nc)
        
    def forward(self, x, z):
        x_s = self.shortcut(x, z)
        dx = self.conv_0(self.actvn(self.norm_0(x, z)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, z)))
        out = x_s + dx
        return out

    def shortcut(self, x, z):
        x_s = self.conv_s(self.actvn(self.norm_s(x, z)))
        return x_s   


class FineEncoder(nn.Module):
    """docstring for Encoder"""
    def __init__(self, image_nc, ngf, img_f, layers, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineEncoder, self).__init__()
        self.layers = layers
        self.first = FirstBlock2d(image_nc, ngf, norm_layer, nonlinearity, use_spect)
        for i in range(layers):
            in_channels = min(ngf*(2**i), img_f)
            out_channels = min(ngf*(2**(i+1)), img_f)
            model = DownBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'down' + str(i), model)
        self.output_nc = out_channels

    def forward(self, x):
        x = self.first(x)
        out=[x]
        for i in range(self.layers):
            model = getattr(self, 'down'+str(i))
            x = model(x)
            out.append(x)
        return out


class FineDecoder(nn.Module):
    """docstring for FineDecoder"""
    def __init__(self, image_nc, feature_nc, ngf, img_f, layers, num_block, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineDecoder, self).__init__()
        self.layers = layers
        for i in range(layers)[::-1]:
            in_channels = min(ngf*(2**(i+1)), img_f)
            out_channels = min(ngf*(2**i), img_f)
            up = UpBlock2d(in_channels, out_channels, norm_layer, nonlinearity, use_spect)
            res = FineADAINResBlocks(num_block, in_channels, feature_nc, norm_layer, nonlinearity, use_spect)
            jump = Jump(out_channels, norm_layer, nonlinearity, use_spect)
            setattr(self, 'up' + str(i), up)
            setattr(self, 'res' + str(i), res)            
            setattr(self, 'jump' + str(i), jump)
        self.final = FinalBlock2d(out_channels, image_nc, use_spect, 'tanh')
        self.output_nc = out_channels

    def forward(self, x, z):
        out = x.pop()
        for i in range(self.layers)[::-1]:
            res_model = getattr(self, 'res' + str(i))
            up_model = getattr(self, 'up' + str(i))
            jump_model = getattr(self, 'jump' + str(i))
            out = res_model(out, z)
            out = up_model(out)
            out = jump_model(x.pop()) + out
        out_image = self.final(out)
        return out_image


class ADAINEncoder(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, layers, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(ADAINEncoder, self).__init__()
        self.layers = layers
        self.input_layer = nn.Conv2d(image_nc, ngf, kernel_size=7, stride=1, padding=3)
        for i in range(layers):
            in_channels = min(ngf * (2**i), img_f)
            out_channels = min(ngf *(2**(i+1)), img_f)
            model = ADAINEncoderBlock(in_channels, out_channels, pose_nc, nonlinearity, use_spect)
            setattr(self, 'encoder' + str(i), model)
        self.output_nc = out_channels
        
    def forward(self, x, z):
        out = self.input_layer(x)
        out_list = [out]
        for i in range(self.layers):
            model = getattr(self, 'encoder' + str(i))
            out = model(out, z)
            out_list.append(out)
        return out_list
        
        
class ADAINDecoder(nn.Module):
    """docstring for ADAINDecoder"""
    def __init__(self, pose_nc, ngf, img_f, encoder_layers, decoder_layers, skip_connect=True, 
                 nonlinearity=nn.LeakyReLU(), use_spect=False):

        super(ADAINDecoder, self).__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.skip_connect = skip_connect
        use_transpose = True
        for i in range(encoder_layers-decoder_layers, encoder_layers)[::-1]:
            in_channels = min(ngf * (2**(i+1)), img_f)
            in_channels = in_channels*2 if i != (encoder_layers-1) and self.skip_connect else in_channels
            out_channels = min(ngf * (2**i), img_f)
            model = ADAINDecoderBlock(in_channels, out_channels, out_channels, pose_nc, use_transpose, nonlinearity, use_spect)
            setattr(self, 'decoder' + str(i), model)
        self.output_nc = out_channels*2 if self.skip_connect else out_channels

    def forward(self, x, z):
        out = x.pop() if self.skip_connect else x
        for i in range(self.encoder_layers-self.decoder_layers, self.encoder_layers)[::-1]:
            model = getattr(self, 'decoder' + str(i))
            out = model(out, z)
            out = torch.cat([out, x.pop()], 1) if self.skip_connect else out
        return out


class ADAINHourglass(nn.Module):
    def __init__(self, image_nc, pose_nc, ngf, img_f, encoder_layers, decoder_layers, nonlinearity, use_spect):
        super(ADAINHourglass, self).__init__()
        self.encoder = ADAINEncoder(image_nc, pose_nc, ngf, img_f, encoder_layers, nonlinearity, use_spect)
        self.decoder = ADAINDecoder(pose_nc, ngf, img_f, encoder_layers, decoder_layers, True, nonlinearity, use_spect)
        self.output_nc = self.decoder.output_nc

    def forward(self, x, z):
        return self.decoder(self.encoder(x, z), z)        


class FineADAINLama(nn.Module):
    def __init__(self, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FineADAINLama, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.actvn = nonlinearity
        ratio_gin = 0.75
        ratio_gout = 0.75        
        self.ffc = FFC(input_nc, input_nc, 3,
                       ratio_gin, ratio_gout, 1, 1, 1,
                       1, False, False, padding_type='reflect')
        global_channels = int(input_nc * ratio_gout)
        self.bn_l = ADAIN(input_nc - global_channels, feature_nc)
        self.bn_g = ADAIN(global_channels, feature_nc)

    def forward(self, x, z):
        x_l, x_g = self.ffc(x)
        x_l = self.actvn(self.bn_l(x_l,z))
        x_g = self.actvn(self.bn_g(x_g,z))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, feature_dim, padding_type='reflect', norm_layer=BatchNorm2d, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FineADAINLama(dim, feature_dim, **conv_kwargs)
        self.conv2 = FineADAINLama(dim, feature_dim, **conv_kwargs)
        self.inline = True

    def forward(self, x, z):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g), z)
        x_l, x_g = self.conv2((x_l, x_g), z)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class FFCADAINResBlocks(nn.Module):
    def __init__(self, num_block, input_nc, feature_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(FFCADAINResBlocks, self).__init__()                                
        self.num_block = num_block
        for i in range(num_block):
            model = FFCResnetBlock(input_nc, feature_nc, norm_layer, nonlinearity, use_spect)
            setattr(self, 'res'+str(i), model)

    def forward(self, x, z):
        for i in range(self.num_block):
            model = getattr(self, 'res'+str(i))
            x = model(x, z)
        return x 


class Jump(nn.Module):
    def __init__(self, input_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.LeakyReLU(), use_spect=False):
        super(Jump, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        conv = spectral_norm(nn.Conv2d(input_nc, input_nc, **kwargs), use_spect)
        if type(norm_layer) == type(None):
            self.model = nn.Sequential(conv, nonlinearity)
        else:
            self.model = nn.Sequential(conv, norm_layer(input_nc), nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out   


class FinalBlock2d(nn.Module):
    def __init__(self, input_nc, output_nc, use_spect=False, tanh_or_sigmoid='tanh'):
        super(FinalBlock2d, self).__init__()
        kwargs = {'kernel_size': 7, 'stride': 1, 'padding':3}
        conv = spectral_norm(nn.Conv2d(input_nc, output_nc, **kwargs), use_spect)
        if tanh_or_sigmoid == 'sigmoid':
            out_nonlinearity = nn.Sigmoid()
        else:
            out_nonlinearity = nn.Tanh()            
        self.model = nn.Sequential(conv, out_nonlinearity)

    def forward(self, x):
        out = self.model(x)
        return out    


class ModulatedConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 num_style_feat,
                 demodulate=True,
                 sample_mode=None,
                 eps=1e-8):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.sample_mode = sample_mode
        self.eps = eps

        # modulation inside each modulated conv
        self.modulation = nn.Linear(num_style_feat, in_channels, bias=True)
        # initialization
        default_init_weights(self.modulation, scale=1, bias_fill=1, a=0, mode='fan_in', nonlinearity='linear')

        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size) /
            math.sqrt(in_channels * kernel_size**2))
        self.padding = kernel_size // 2

    def forward(self, x, style):
        b, c, h, w = x.shape   
        style = self.modulation(style).view(b, 1, c, 1, 1)
        weight = self.weight * style  

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(b, self.out_channels, 1, 1, 1)

        weight = weight.view(b * self.out_channels, c, self.kernel_size, self.kernel_size)

        # upsample or downsample if necessary
        if self.sample_mode == 'upsample':
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        elif self.sample_mode == 'downsample':
            x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        out = F.conv2d(x, weight, padding=self.padding, groups=b)
        out = out.view(b, self.out_channels, *out.shape[2:4])
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size}, demodulate={self.demodulate}, sample_mode={self.sample_mode})')


class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_style_feat, demodulate=True, sample_mode=None):
        super(StyleConv, self).__init__()
        self.modulated_conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, num_style_feat, demodulate=demodulate, sample_mode=sample_mode)
        self.weight = nn.Parameter(torch.zeros(1))  # for noise injection
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, style, noise=None):
        # modulate
        out = self.modulated_conv(x, style) * 2**0.5  # for conversion
        # noise injection
        if noise is None:
            b, _, h, w = out.shape
            noise = out.new_empty(b, 1, h, w).normal_()
        out = out + self.weight * noise
        # add bias
        out = out + self.bias
        # activation
        out = self.activate(out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channels, num_style_feat, upsample=True):
        super(ToRGB, self).__init__()
        self.upsample = upsample
        self.modulated_conv = ModulatedConv2d(
            in_channels, 3, kernel_size=1, num_style_feat=num_style_feat, demodulate=False, sample_mode=None)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.modulated_conv(x, style)
        out = out + self.bias
        if skip is not None:
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='bilinear', align_corners=False)
            out = out + skip
        return out
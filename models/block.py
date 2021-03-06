from collections import OrderedDict
import torch
import torch.nn as nn
class OperateBlock(nn.Module):

    def __init__(self, submodule):
        super(OperateBlock, self).__init__()
        self.sub = submodule
    def forward(self, x):
        result = x + self.sub(x)
        return result
def BlockSequent(*args):
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)
def conv_block(inputNC, outputNC, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, activation='relu', mode='CNA'):

    kernel_size_t = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size_t - 1) // 2
    p = None if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(inputNC, outputNC, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = None
    if activation=='relu':
        a = nn.relu(True)
    elif activation=='leakyrelu':
        a = nn.LeakyReLU(True)
    n = nn.BatchNorm2d(outputNC, affine=True) if norm_type else None
    return BlockSequent(p, c, n, a)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, activation='leakyrelu', mode='CNA'):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, activation=activation, mode=mode)
        self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, activation=activation, mode=mode)
        self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, activation=activation, mode=mode)
        self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, activation=activation, mode=mode)
        self.conv5 = conv_block(nc+4*gc, nc, 3, stride, bias=bias, pad_type=pad_type, \
            norm_type=norm_type, activation=None, mode=mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x
class RRDB(nn.Module):
    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=None, activation='leakyrelu', mode='CNA'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, activation, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, activation, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias, pad_type, \
            norm_type, activation, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(0.2) + x

def pixelshuffle_block(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                        pad_type='zero', norm_type=None, activation='relu'):
    conv = conv_block(in_nc, out_nc * (upscale_factor ** 2), kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=None, activation=None)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = nn.BatchNorm2d(out_nc, affine=True) if norm_type else None
    a = None
    if activation=='relu':
        a = nn.relu(True)
    elif activation=='leakyrelu':
        a = nn.LeakyReLU(True)
    return BlockSequent(conv, pixel_shuffle, n, a)
def upconv_blcok(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, bias=True, \
                pad_type='zero', norm_type=None, activation='relu', mode='nearest'):

    upsample = nn.Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(in_nc, out_nc, kernel_size, stride, bias=bias, \
                        pad_type=pad_type, norm_type=norm_type, activation=activation)
    return BlockSequent(upsample, conv)

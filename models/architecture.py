import math
import torch
import torch.nn as nn
import torchvision
from . import block as B

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            activation='leakyrelu', mode='CNA'):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, activation=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, activation=activation, mode='CNA') for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, activation=None, mode=mode)
        upsample_block = B.upconv_blcok

        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, activation=activation)
        else:
            upsampler = [upsample_block(nf, nf, activation=activation) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, activation=activation)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, activation=None)

        self.model = B.BlockSequent(fea_conv, B.OperateBlock(B.BlockSequent(*rb_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', activation='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_128, self).__init__()

        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, activation=activation, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            activation=activation, mode=mode)

        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            activation=activation, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            activation=activation, mode=mode)

        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            activation=activation, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            activation=activation, mode=mode)

        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            activation=activation, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            activation=activation, mode=mode)

        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            activation=activation, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            activation=activation, mode=mode)

        self.features = B.BlockSequent(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGGFeatureExtractor(nn.Module):
    def __init__(self,
                 feature_layer=34,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)

        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        output = self.features(x)
        return output

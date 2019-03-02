import functools
import torch
import torch.nn as nn
from torch.nn import init
import models.architecture as arch
def weights_by(scale=1):
    def helper(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)
    return helper
def define_G(opt):
    opt_net = opt['network_G']
    netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'])

    if opt['is_train']:
        netG.apply(weights_by(0.1))
    netG = nn.DataParallel(netG)
    return netG
def define_D(opt):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']

    netD = arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])

    netD.apply(weights_by(1))
    netD = nn.DataParallel(netD)
    return netD
def define_F(opt):

    netF = arch.VGGFeatureExtractor(feature_layer=34, \
        device=torch.device('cuda'))
    netF = nn.DataParallel(netF)
    netF.eval()
    return netF

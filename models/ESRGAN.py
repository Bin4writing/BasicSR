from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import init
import functools
from models.architecture import VGGFeatureExtractor,RRDBNet,Discriminator_VGG_128

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

class ESRGAN():
    def __init__(self, cnf):
        self.cnf = cnf
        self.device = torch.device('cuda')
        self.name = cnf['name']
        self.is_train = cnf['is_train']
        self.schedulers = []
        self.optimizers = []

        self.GAN = RRDBNet(in_nc=3, out_nc=3, nf=64,
            nb=23, gc=32, upscale=4, norm_type=None,
            activation='leakyrelu', mode='CNA')

        if cnf['is_train']:
            self.GAN.apply(weights_by(0.1))
        self.GAN = nn.DataParallel(self.GAN).to(self.device)
        if self.is_train:
            self.discriminator =Discriminator_VGG_128(in_nc=3, base_nf=64, \
                norm_type='batch', mode='CNA', activation='leakyrelu')
            self.discriminator.apply(weights_by(1))
            self.discriminator = nn.DataParallel(self.discriminator).to(self.device)
            self.GAN.train()
            self.discriminator.train()
        self.load()

        if self.is_train:
            self.cri_pix = nn.L1Loss().to(self.device)
            self.cri_fea = nn.L1Loss().to(self.device)
            self.feature_extractor = VGGFeatureExtractor(feature_layer=34, \
                device=self.device)
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
            self.feature_extractor.eval()
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.cri_gan = nn.BCEWithLogitsLoss().to(self.device)
            self.l_gan_w = self.cnf['w_gan']
            self.D_update_ratio = self.cnf['D_update_ratio'] if self.cnf['D_update_ratio'] else 1
            self.D_init_iters = self.cnf['D_init_iters'] if self.cnf['D_init_iters'] else 0

            wd_G = self.cnf['weight_decay_G'] if self.cnf['weight_decay_G'] else 0
            opp = []
            for k, v in self.GAN.named_parameters():
                if v.requires_grad:
                    opp.append(v)
            self.optimizer_G = torch.optim.Adam(opp, lr=self.cnf['lr_G'], \
                weight_decay=0, betas=(self.cnf['betaG'], 0.999))
            self.optimizers.append(self.optimizer_G)

            wd_D = self.cnf['weight_decay_D'] if self.cnf['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.cnf['lr_D'], \
                weight_decay=0, betas=(self.cnf['betaD'], 0.999))
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                    self.cnf['dec_lr_points'], self.cnf['lr_gamma']))

            self.log_dict = OrderedDict()
            self.w_pix = self.cnf['w_pix']
            self.w_fea = self.cnf['w_fea']
    def feed_data(self, data):
        self.var_L = data['LR'].to(self.device)
        self.var_H = data['HR'].to(self.device)
    def optimize_parameters(self, step):
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.GAN(self.var_L)
        minibatch = self.self.cnf['minibatch']
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_g_pix = self.w_pix * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
            real_fea = self.feature_extractor(self.var_H).detach()
            fake_fea = self.feature_extractor(self.fake_H)
            l_g_fea = self.w_fea*self.cri_fea(fake_fea,real_fea)

            d_fake = self.discriminator(self.fake_H)
            d_real = self.discriminator(self.var_H).detach()

            l_g_gan = self.l_gan_w * (self.cri_gan(d_real - torch.mean(d_fake), torch.zeros_like(d_real)) +
                                      self.cri_gan(d_fake - torch.mean(d_real), torch.ones_like(d_fake))) / 2
            l_g_total += l_g_gan
            l_g_total.backward()
            self.optimizer_G.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        d_real = self.discriminator(self.var_H)
        d_fake = self.discriminator(self.fake_H).detach()
        l_d_real = self.cri_gan(d_real - torch.mean(d_fake), torch.ones_like(d_real))
        l_d_fake = self.cri_gan(d_fake - torch.mean(d_real), torch.zeros_like(d_fake))

        l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

    def generate(self,data):
        self.var_L = data['LR'].to(self.device)
        self.GAN.eval()
        with torch.no_grad():
            fake_H = self.GAN(self.var_L)
            return fake_H.detach()[0].float().cpu()

    def test(self):
        self.GAN.eval()
        with torch.no_grad():
            self.fake_H = self.GAN(self.var_L)
        self.GAN.train()

    def get_current_log(self):
        return self.log_dict
    def load(self):
        load_path_G = self.cnf['GAN']['path']
        if load_path_G is not None:
            self.load_network(load_path_G, self.GAN)
        load_path_D = self.cnf['Discriminator']['path']
        if self.cnf['is_train'] and load_path_D is not None:
            self.load_network(load_path_D, self.discriminator)

    def save(self, iter_step):
        self.save_network(self.GAN, 'G', iter_step)
        self.save_network(self.discriminator, 'D', iter_step)

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_learning_rate(self):
        return self.schedulers[0].get_lr()[0]

    def get_Discriminatorescription(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_step):
        save_filename = '{}_{}.pth'.format(iter_step, network_label)
        save_path = os.path.join(self.cnf['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(load_path), strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.cnf['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']


        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

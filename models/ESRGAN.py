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
        train_cnf = cnf['train']
        self.train_cnf = train_cnf

        cnf_net = cnf['network_G']
        self.GAN = RRDBNet(in_nc=cnf_net['in_nc'], out_nc=cnf_net['out_nc'], nf=cnf_net['nf'],
            nb=cnf_net['nb'], gc=cnf_net['gc'], upscale=cnf_net['scale'], norm_type=cnf_net['norm_type'],
            act_type='leakyrelu', mode=cnf_net['mode'])

        if cnf['is_train']:
            self.GAN.apply(weights_by(0.1))
        self.GAN = nn.DataParallel(self.GAN).to(self.device)
        if self.is_train:
            cnf_net = cnf['network_D']
            self.discriminator =Discriminator_VGG_128(in_nc=cnf_net['in_nc'], base_nf=cnf_net['nf'], \
                norm_type=cnf_net['norm_type'], mode=cnf_net['mode'], act_type=cnf_net['act_type'])
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
            self.l_gan_w = train_cnf['gan_weight']
            self.D_update_ratio = train_cnf['D_update_ratio'] if train_cnf['D_update_ratio'] else 1
            self.D_init_iters = train_cnf['D_init_iters'] if train_cnf['D_init_iters'] else 0

            wd_G = train_cnf['weight_decay_G'] if train_cnf['weight_decay_G'] else 0
            opp = []
            for k, v in self.GAN.named_parameters():
                if v.requires_grad:
                    opp.append(v)
            self.optimizer_G = torch.cnfim.Adam(opp, lr=train_cnf['lr_G'], \
                weight_decay=wd_G, betas=(train_cnf['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)

            wd_D = train_cnf['weight_decay_D'] if train_cnf['weight_decay_D'] else 0
            self.optimizer_D = torch.cnfim.Adam(self.discriminator.parameters(), lr=train_cnf['lr_D'], \
                weight_decay=wd_D, betas=(train_cnf['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                    train_cnf['lr_steps'], train_cnf['lr_gamma']))

            self.log_dict = OrderedDict()
            self.l_pix_w = train_cnf['pixel_weight']
            self.l_fea_w = train_cnf['feature_weight']
    def feed_data(self, data):
        self.var_L = data['LR'].to(self.device)
        self.var_H = data['HR'].to(self.device)
    def cnfimize_parameters(self, step):
        for p in self.discriminator.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.GAN(self.var_L)
        batch_size = self.train_cnf['batch_size']
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
            real_fea = self.feature_extractor(self.var_H).detach()
            fake_fea = self.feature_extractor(self.fake_H)
            l_g_fea = self.l_fea_w*self.cri_fea(fake_fea,real_fea)

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
        load_path_G = self.cnf['path']['pretrain_model_G']
        if load_path_G is not None:
            self.load_network(load_path_G, self.GAN)
        load_path_D = self.cnf['path']['pretrain_model_D']
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

    def get_network_description(self, network):
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
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

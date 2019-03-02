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
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.name = opt['name']
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        train_opt = opt['train']
        self.train_opt = train_opt
        # define networks and load pretrained models
        opt_net = opt['network_G']
        self.netG = RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'])

        if opt['is_train']:
            self.netG.apply(weights_by(0.1))
        self.netG = nn.DataParallel(self.netG).to(self.device)
        if self.is_train:
            opt_net = opt['network_D']
            netD =Discriminator_VGG_128(in_nc=opt_net['in_nc'], base_nf=opt_net['nf'], \
                norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
            netD.apply(weights_by(1))
            netD = nn.DataParallel(self.netD).to(self.device)
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            self.cri_pix = nn.L1Loss().to(self.device)
            self.cri_fea = nn.L1Loss().to(self.device)
            self.netF = VGGFeatureExtractor(feature_layer=34, \
                device=self.device)
            self.netF = nn.DataParallel(self.netF)
            self.netF.eval()
            self.netF = self.netF.to(self.device)
            self.cri_gan = nn.BCEWithLogitsLoss().to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            for optimizer in self.optimizers:
                self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                    train_opt['lr_steps'], train_opt['lr_gamma']))

            self.log_dict = OrderedDict()
            self.l_pix_w = train_opt['pixel_weight']
            self.l_fea_w = train_opt['feature_weight']
    def feed_data(self, data):
        self.var_L = data['LR'].to(self.device)

        self.var_H = data['HR'].to(self.device)
    def optimize_parameters(self, step):
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)
        batch_size = self.train_opt['batch_size']
        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
            l_g_total += l_g_pix
            real_fea = self.netF(self.var_H).detach()
            fake_fea = self.netF(self.fake_H)
            l_g_fea = self.l_fea_w*self.cri_fea(fake_fea,real_fea)

            # G gan + cls loss
            d_fake = self.netD(self.fake_H)
            d_real = self.netD(self.var_H).detach()

            l_g_gan = self.l_gan_w * (self.cri_gan(d_real - torch.mean(d_fake), torch.zeros_like(d_real)) +
                                      self.cri_gan(d_fake - torch.mean(d_real), torch.ones_like(d_fake))) / 2
            l_g_total += l_g_gan
            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        d_real = self.netD(self.var_H)
        d_fake = self.netD(self.fake_H).detach()  # detach to avoid BP to G
        l_d_real = self.cri_gan(d_real - torch.mean(d_fake), torch.ones_like(d_real))
        l_d_fake = self.cri_gan(d_fake - torch.mean(d_real), torch.zeros_like(d_fake))

        l_d_total = (l_d_real + l_d_fake) / 2

        l_d_total.backward()
        self.optimizer_D.step()

    def generate(self,data):
        self.var_L = data['LR'].to(self.device)
        self.netG.eval()
        with torch.no_grad():
            fake_H = self.netG(self.var_L)
            return fake_H.detach()[0].float().cpu()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        return self.log_dict
    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            self.load_network(load_path_G, self.netG)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            self.load_network(load_path_D, self.netD)

    def save(self, iter_step):
        self.save_network(self.netG, 'G', iter_step)
        self.save_network(self.netD, 'D', iter_step)

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
        save_path = os.path.join(self.opt['path']['models'], save_filename)
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
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
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


import torch
import torch.nn as nn


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class ModelLoss(nn.Module):
    def __init__(self,model):
        super(ModelLoss,self).__init__()
        self.model = model

    def forward(that):
        self = that.model
        l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
        l_g_total += l_g_pix
        real_fea = self.netF(self.var_H).detach()
        fake_fea = self.netF(self.fake_H)
        l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
        l_g_total += l_g_fea          
        # G gan + cls loss
        pred_g_fake = self.netD(self.fake_H)
        pred_d_real = self.netD(self.var_ref).detach()

        l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                  self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total += l_g_gan
        return l_g_total
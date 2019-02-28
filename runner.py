import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
from models.ESRGAN import ESRGAN

import torch

from config import Config

import util
from data import Dataset

import yaml


class Runner:
    def __init__(self, path,tf_log_dir='./tf_log/'):
        self.config = Config(path).load().setMissingNone()
        self._path = path
        self._tf_logger = None 
        self.model = None
        self._tf_log_dir = tf_log_dir

        self.current_step = 0
        self.start_epoch = 0 
        self.total_epochs = 0
        self.total_iters = 0
        self.datasets = []
        self.val_datasets = []

        self.resume_state = None

        self.checkResume()
    
    @property
    def tf_logger(self):
        if not self._tf_logger:
            from tensorboardX import SummaryWriter
            self._tf_logger = SummaryWriter(log_dir= self._tf_log_dir + self.config['name'])
        return self._tf_logger
    
    @property
    def path(self):
        return self._path 

    @path.setter 
    def path(self,val):
        self.config.load(path=val).setMissingNone()
        self._path = val 

    
    def log(self,msg):
        print(util.get_timestamp()+': '+msg)

    def prepare(self):
        raise NotImplementedError('must implement prepare!')

    def run(self):
        raise NotImplementedError('must implement run!')
    
    def __str__(self):
        return 'current_step: {}, start_epoch: {}, total_epochs: {}, total_iters: {}'.format(self.current_step,self.start_epoch,self.total_epochs,self.total_iters)

    def checkResume(self):
        if self.config['path']['resume_state']:
            self.resume_state = torch.load(self.config['path']['resume_state'])
            self.log('Resuming training from epoch: {}, iter: {}.'.format(
                self.resume_state['epoch'], self.resume_state['iter']))
            self.config.checkResume()
            self.start_epoch = self.resume_state['epoch']
            self.current_step = self.resume_state['iter']
        else:
            util.mkdir_and_rename(self.config['path']['experiments_root'])  # rename old folder if exists
            util.mkdirs((path for key, path in self.config['path'].items() if not key == 'experiments_root'
                        and 'pretrain_model' not in key and 'resume' not in key))

    def clear(self):
        self.datasets = []
        self.val_datasets = []
        self.current_step = 0
        self.start_epoch = 0 
        self.total_epochs = 0
        self.total_iters = 0
        self.resume_state = None

class TrainRunner(Runner):
    def __init__(self, path='config/train.yaml'):
        super().__init__(path)

    def prepare(self):
        seed = self.config['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        self.log('Random seed: {}'.format(seed))
        util.set_random_seed(seed)

        torch.backends.cudnn.benchmark = True

        for cnf in self.config['datasets'].values():
            ds = Dataset(cnf).createLoader()
            if ds.phase == 'train':
                train_size = int(math.ceil(len(ds) / cnf['batch_size']))
                self.log('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(ds), train_size))

                self.total_iters = int(self.config['train']['niter'])
                self.total_epochs = int(math.ceil(self.total_iters / train_size))
                
                self.log('Total epochs needed: {:d} for iters {:,d}'.format(
                    self.total_epochs, self.total_iters))
                self.datasets.append(ds)
            elif ds.phase == 'val':
                self.val_datasets.append(ds)
                self.log('Number of val images in [{:s}]: {:d}'.format(cnf['name'],
                                                                    len(ds)))
        self.model = ESRGAN(self.config)

    def run(self):
        assert self.model is not None, "model not created"

        if self.start_epoch: self.model.resume_training(self.resume_state)
        self.log('Start training from epoch: {:d}, iter: {:d}'.format(self.start_epoch, self.current_step))
        for epoch in range(self.start_epoch, self.total_epochs):
            for ds in self.datasets:
                for data in ds.loader:
                    self.current_step += 1
                    if self.current_step > self.total_iters: break
                    
                    self.model.update_learning_rate()

                    self.model.feed_data(data)
                    self.model.optimize_parameters(self.current_step)

                    # log
                    if self.current_step % self.config['logger']['print_freq'] == 0:
                        logs = self.model.get_current_log()
                        message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                            epoch, self.current_step, self.model.get_current_learning_rate())
                        for k, v in logs.items():
                            message += '{:s}: {:.4e} '.format(k, v)
                            if self.config['use_tb_logger'] and 'debug' not in self.config['name']:
                                self.tf_logger.add_scalar(k, v, self.current_step)
                        self.log(message)

                    # validation
                    if self.current_step % self.config['train']['val_freq'] == 0:
                        avg_psnr = 0.0
                        idx = 0
                        for val_ds in self.val_datasets:
                            for val_data in val_ds.loader:
                                idx += 1
                                img_name = os.path.splitext(os.path.basename(val_ds['LR_path'][0]))[0]
                                img_dir = os.path.join(self.config['path']['val_images'], img_name)
                                util.mkdir(img_dir)

                                self.model.feed_data(val_ds.loader)
                                self.model.test()

                                visuals = self.model.get_current_visuals()
                                sr_img = util.tensor2img(visuals['SR'])  # uint8
                                gt_img = util.tensor2img(visuals['HR'])  # uint8


                                # calculate PSNR
                                crop_size = self.config['scale']
                                gt_img = gt_img / 255.
                                sr_img = sr_img / 255.
                                cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                                cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                                avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)

                        avg_psnr = avg_psnr / idx

                        self.log('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                        self.log('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                            epoch, self.current_step, avg_psnr))

                        if self.config['use_tb_logger'] and 'debug' not in self.config['name']:
                            self.tf_logger.add_scalar('psnr', avg_psnr, self.current_step)

                    # save models and training states
                    if self.current_step % self.config['logger']['save_checkpoint_freq'] == 0:
                        self.log('Saving models and training states.')
                        self.model.save(self.current_step)
                        self.model.save_training_state(epoch, self.current_step)

        self.log('Saving the final self.model.')
        self.model.save('latest')
        self.log('End of training.')



if __name__ == "__main__":
    train_runner = TrainRunner()
    train_runner.prepare()
    train_runner.run()

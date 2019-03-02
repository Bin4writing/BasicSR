import torch.utils.data

import os.path
import random
import numpy as np
import cv2
import torch.utils as torch_util
from torch import from_numpy
from torch.utils.data import Dataset as TorchDataset
import torch.utils.data as data
import data.util as util
class Dataset(TorchDataset):

    def __init__(self, cnf):
        super(Dataset, self).__init__()
        self.phase = cnf['phase']
        self.cnf = cnf
        self.name = cnf['name']

        self.HR_env, self.paths_HR = util.get_image_paths(cnf['data_type'], cnf['dataroot_HR'])
        self.LR_env, self.paths_LR = util.get_image_paths(cnf['data_type'], cnf['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]
    def createLoader(self):
        self.loader = torch_util.data.DataLoader(
            self,
            batch_size=self.cnf['batch_size'],
            shuffle=self.cnf['use_shuffle'],
            num_workers=self.cnf['n_workers'],
            drop_last=True,
            pin_memory=True) if self.phase == 'train' else torch_util.data.DataLoader(
            self, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        return self

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.cnf['scale']
        HR_size = self.cnf['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)
        # modcrop in the validation / test phase
        if self.cnf['phase'] != 'train':
            img_HR = util.modcrop(img_HR, scale)
        # change color space if necessary
        if self.cnf['color']:
            img_HR = util.channel_convert(img_HR.shape[2], self.cnf['color'], [img_HR])[0]

        # get LR image
        LR_path = self.paths_LR[index]
        img_LR = util.read_img(self.LR_env, LR_path)

        if self.cnf['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_HR.shape
            if H < HR_size or W < HR_size:
                img_HR = cv2.resize(
                    np.copy(img_HR), (HR_size, HR_size), interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LR = util.imresize_np(img_HR, 1 / scale, True)
                if img_LR.ndim == 2:
                    img_LR = np.expand_dims(img_LR, axis=2)

            H, W, C = img_LR.shape
            LR_size = HR_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LR_size))
            rnd_w = random.randint(0, max(0, W - LR_size))
            img_LR = img_LR[rnd_h:rnd_h + LR_size, rnd_w:rnd_w + LR_size, :]
            rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
            img_HR = img_HR[rnd_h_HR:rnd_h_HR + HR_size, rnd_w_HR:rnd_w_HR + HR_size, :]

            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.cnf['use_flip'], \
                self.cnf['use_rot'])

        # change color space if necessary
        if self.cnf['color']:
            img_LR = util.channel_convert(C, self.cnf['color'], [img_LR])[0] # TODO during val no definetion

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)


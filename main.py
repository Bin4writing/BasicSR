import os
import subprocess
try:
    import gnureadline
    import sys
    sys.modules['readline'] = gnureadline
except ImportError:
    pass
import cmd

import argparse
import sys
import os.path
from multiprocessing import Pool
import glob
import pickle
import lmdb
import cv2
import random
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.util import *

from runner import TrainRunner
from generate import Generator

def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)

class Dataset:
    def __init__(self, root=os.path.join(os.environ['HOME'],'datasets')):
        self.root = root
        self.img_folder = ''
        self.img_sub_folder = '_sub'
        self.img_sub_bicLRx4_folder = '_bicLRx4'
        self._name = ''

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,val):
        self._name = val
        self.img_folder = os.path.join(self.root,val)
        self.img_sub_folder = self.img_folder + '_sub'
        self.img_sub_bicLRx4_folder = self.img_sub_folder + '_bicLRx4'

    def lmdb(self,type_str):
        img_list = []
        lmdb_save_path = ''
        if type_str=='hr':
            img_list.extend(sorted(glob.glob(self.img_folder+'/*')))
            lmdb_save_path = self.img_folder + '.lmdb'
        elif type_str=='sub':
            img_list.extend(sorted(glob.glob(self.img_sub_folder+'/*')))
            lmdb_save_path = self.img_sub_folder + '.lmdb'
        elif type_str=='lr':
            img_list.extend(sorted(glob.glob(self.img_sub_bicLRx4_folder+'/*')))
            lmdb_save_path = self.img_sub_bicLRx4_folder + '.lmdb'
        else:
            print('type not recognized!')
            return
        dataset = []
        data_size = 0

        print('Read images...')
        for i, v in enumerate(img_list):
            img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
            dataset.append(img)
            data_size += img.nbytes
        env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
        print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

        with env.begin(write=True) as txn:  # txn is a Transaction object
            for i, v in enumerate(img_list):
                base_name = os.path.splitext(os.path.basename(v))[0]
                key = base_name.encode('ascii')
                data = dataset[i]
                if dataset[i].ndim == 2:
                    H, W = dataset[i].shape
                    C = 1
                else:
                    H, W, C = dataset[i].shape
                meta_key = (base_name + '.meta').encode('ascii')
                meta = '{:d}, {:d}, {:d}'.format(H, W, C)

                txn.put(key, data)
                txn.put(meta_key, meta.encode('ascii'))
        print('Finish writing lmdb.')

        keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
        env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            print('Create lmdb keys cache: {}'.format(keys_cache_file))
            keys = [key.decode('ascii') for key, _ in txn.cursor()]
            pickle.dump(keys, open(keys_cache_file, "wb"))
        print('Finish creating lmdb keys cache.')
        return self
    def sub(self,count=800,crop_sz=480):
        n_thread = 20
        step = 240
        thres_sz = 48
        compression_level = 3
        if not os.path.exists(self.img_sub_folder):
            os.makedirs(self.img_sub_folder)
            print('mkdir [{:s}] ...'.format(self.img_sub_folder))
        else:
            print('Folder [{:s}] already exists. Exit...'.format(self.img_sub_folder))
            sys.exit(1)

        img_list = []
        for root, _, file_list in sorted(os.walk(self.img_folder)):
            path = [os.path.join(root, x) for x in file_list]  # assume only images in the self.img_folder
            if count < len(path):
                img_list.extend(path[:count])
                count = 0
                break
            else:
                count -= len(path)
                img_list.extend(path)

        def update(arg):
            return
        pool = Pool(n_thread)
        for path in img_list:
            pool.apply_async(worker,
                args=(path, self.img_sub_folder, crop_sz, step, thres_sz, compression_level),
                callback=update)
        pool.close()
        pool.join()
        print('All subprocesses done.')
        return self

    def bicLRx4(self,scale=4):
        if not os.path.exists(self.img_sub_bicLRx4_folder):
            os.makedirs(self.img_sub_bicLRx4_folder)
        items = os.listdir(self.img_sub_folder)
        print(len(items))
        for item in items:
            if not item.endswith('.png'):
                continue
            in_file = os.path.join(self.img_sub_folder, item)
            out_file = os.path.join(self.img_sub_bicLRx4_folder, item)
            img = cv2.imread(in_file)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()

            rlt = imresize(img, 1/scale, antialiasing=True)

            import torchvision.utils

            torchvision.utils.save_image(
                (rlt * 255).round() / 255, out_file, nrow=1, padding=0, normalize=False)
        return self
class Console(cmd.Cmd):
    def __init__(self):
        super().__init__()
        self.dataset = Dataset()
        self.runner = TrainRunner()
        self.generator = Generator()
        self.generate_parser = argparse.ArgumentParser()
        self.generate_parser.add_argument("--count", help="set count of HR imgs to be used, default to the total", type=int)
        self.generate_parser.add_argument("--size", help="set size of a sub-crop img, default to 480", type=int)

    def do_root(self,path):
        self.dataset.root = os.path.expanduser(path)
        self.generator.root = self.dataset.root
        print('dataset root dir has been set to {}'.format(self.dataset.root))

    def do_load(self,name_str):
        name_str = name_str.strip()
        if not name_str:
            self.help_load()
            return
        self.dataset.name = name_str
        self.generator.name = name_str
        train_conf = self.runner.config['datasets']['train']
        train_conf['dataroot_HR'] = self.dataset.img_sub_folder + '.lmdb'
        train_conf['dataroot_LR'] = self.dataset.img_sub_bicLRx4_folder + '.lmdb'

    def help_load(self):
        print('\n'.join([
            'load <name>',
            'load from named subdir of {}'.format(self.dataset.root)
            ]))

    def do_generate(self,args):
        arg_lst = args.strip().split(' ')
        arg_lst = [ i for i in arg_lst if i]
        args = self.generate_parser.parse_args(arg_lst)
        conf_sub = {}
        if args.count:
            conf_sub['count'] = args.count
        if args.size:
            conf_sub['crop_sz'] = args.size
        conf_bicLRx4 = {}
        self.dataset.sub(**conf_sub).bicLRx4(**conf_bicLRx4).lmdb('sub').lmdb('lr')
    def help_generate(self):
        print('\n'.join([
            'todo'
            ]))
    def do_sub(self,size_str):
        size_str = size_str.strip()
        if not size_str:
            self.dataset.sub()
        else:
            self.dataset.sub(crop_sz=int(size_str))
    def do_recover(self,path):
        path = path.strip()
        if not path:
            self.generator.prepare().run()
        self.generator.path = path
        self.generator.clear().prepare().run()

    def help_recover(self):
        print('\n'.join([
            'recover HR from LR'
            ]))
    def do_lmdb(self,type_str):
        type_str = type_str.strip()
        if not type_str:
            self.help_lmdb()
            return
        self.dataset.lmdb(type_str)
        return True

    def help_lmdb(self):
        print('\n'.join([
            'lmdb <type_str>',
            'enum type_str = ["hr", "sub", "lr"]'
            ]))
    def do_rm(self,type_str):
        type_str = type_str.strip()
        if not type_str:
            self.help_rm()
            return
        torm = ''
        if type_str=='sub':
            torm = self.dataset.img_sub_folder
        elif type_str=='lr':
            torm = self.dataset.img_sub_bicLRx4_folder
        else:
            self.help_rm()
            return
        sub_cmd = subprocess.Popen('rm -rf {} {}'.format(torm,torm+'.lmdb'),shell=True,stdout=subprocess.PIPE)
        output = sub_cmd.communicate()[0].decode('utf-8')
        print(output)
    def help_rm(self):
        print('\n'.join(
            [ 'rm <type_str>','enum type_str= ["sub","lr"]'  ]
            ))
    def do_exit(self,line):
        print('Bye')
        sys.exit(1)

    def do_runit(self,type_str):
        type_str = type_str.strip()
        if not type_str:
            self.help_run()
            return
        if type_str == 'train':
            self.runner.prepare()
            self.runner.run()
        elif type_str == 'test':
            pass
        else:
            pass
    def help_run(self):
        print('\n'.join([
            'run <type_str>',
            'enum type_str = ["train","test","gen"]'
            ]))

    def do_EOF(self, line):
        return True
if __name__ == '__main__':
    Console().cmdloop()

from config import Config 


import os
import util as util
from models.SRRaGAN_model import SRRaGANModel
from data import Dataset




def bak():
    generate_conf = self.config['datasets']['generate']
    self.result = self.config['path']['results_root']
    self._lr_dir = generate_conf['dataroot_LR']
    self._name = generate_conf['dataroot_LR'].split('/')[-1]

class Generator():
    def __init__(self,path='config/generate.yaml'):
        self.config = Config(path).load(is_train=False).setMissingNone()
        self._path = path
        self._model = None
        self.result_dir = self.config['path']['results_root']
        self.datasets = []

    @property 
    def path(self):
        return self._path
    @path.setter 
    def path(self,val):
        self._path = val
        self.config.load(is_train=False,path=val).setMissingNone()
        self.result_dir = self.config['path']['result_root']

    def model_with(self,conf):
        return SRRaGANModel(conf)

    @property
    def model(self):
        if not self._model:
            self._model = self.model_with(self.config)
        return self._model

    def log(self,msg):
        print(msg)

    def prepare(self):
        for cnf in sorted(self.config['datasets'].values()):
            self.log(cnf)
            ds = Dataset(cnf)
            self.log('Number of images in [{:s}]: {:d}'.format(ds.paths_HR, len(ds)))
            self.datasets.append(ds)
        return self

    def clear(self):
        self.datasets = []
        self._model = None
        return self

    def run(self):
        for ds in self.datasets:
            self.log('\nGenerating from [{:s}]...'.format(self.ds.paths_HR))
            ds_dir = os.path.join(self.result_dir,self.model.name,ds.name)
            util.mkdir(ds_dir)
            for data in ds.loader:
                img_path = data['LR_path'][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                sr_img = util.tensor2img(self.model.generate(data))

                suffix = self.config['suffix']
                if suffix:
                    save_img_path = os.path.join(ds_dir, img_name + suffix + '.png')
                else:
                    save_img_path = os.path.join(ds_dir, img_name + '.png')
                util.save_img(sr_img, save_img_path)
        return self


if __name__ == '__main__':
    generator = Generator().prepare().run()
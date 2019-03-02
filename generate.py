from config import Config
import os
import util as util
from models.ESRGAN import ESRGAN
from data import Dataset

class Generator():
    def __init__(self,path='config/generate.yaml'):
        self.config = Config(path).load(is_train=False).setMissingNone()
        self._path = path
        self._model = None
        self.result_dir = self.config['result_dir']
        self.datasets = []

    @property
    def path(self):
        return self._path
    @path.setter
    def path(self,val):
        self._path = val
        self.config.load(is_train=False,path=val).setMissingNone()
        self.result_dir = self.config['result_dir']

    def model_with(self,conf):
        return ESRGAN(conf)

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
            ds = Dataset(cnf).createLoader()
            self.log('Number of images in [{:s}]: {:d}'.format(ds.cnf['lr_dir'], len(ds)))
            self.datasets.append(ds)
        return self

    def clear(self):
        self.datasets = []
        self._model = None
        return self

    def run(self):
        for ds in self.datasets:
            self.log('\nGenerating from [{:s}]...'.format(ds.cnf['lr_dir']))
            ds_dir = os.path.join(self.result_dir,ds.name)

            util.mkdir(ds_dir)
            for data in ds.loader:
                img_path = data['LR_path'][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                sr_img = util.tensor2img(self.model.generate(data))


                save_img_path = os.path.join(ds_dir, img_name + '.png')
                util.save_img(sr_img, save_img_path)
        return self
if __name__ == '__main__':
    generator = Generator().prepare().run()

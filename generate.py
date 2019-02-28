from config import Config 


import os
import util as util
from data import create_dataset, create_dataloader
from models.SRRaGAN_model import SRRaGANModel



class Generator():
    def __init__(self,path='config/generate.yaml'):
        self.config = Config(path).load(False).setMissingNone()
        self._model = None
        self.root = ''
        self.result = ''
        self._lr_dir = ''
        self._name = ''

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self,val):
        self._lr_dir = os.path.join(self.root,val)
        self._name = val
        self.config['datasets']['generate']['dataroot_LR'] = self._lr_dir
        self.config['datasets']['generate']['name'] = val


    @property
    def model(self):
        if not self._model:
            self._model = SRRaGANModel(self.config)
        return self._model

    def log(self,msg):
        print(msg)

    def run(self):
        data_loaders = []
        for phase, dataset_opt in sorted(self.config['datasets'].items()):
            logger.info(dataset_opt)
            data_set = create_dataset(dataset_opt)
            data_loader = create_dataloader(data_set, dataset_opt)
            logger.info('Number of images in [{:s}]: {:d}'.format(self._name, len(data_set)))
            data_loaders.append(data_loader)

        for data_loader in data_loaders:
            data_set_name = self._name
            logger.info('\nGenerating from [{:s}]...'.format(data_set_name))
            dataset_dir = os.path.join(self.result, data_set_name)
            util.mkdir(dataset_dir)

            for data in data_loader:
                model.feed_data(data, need_HR=False)
                img_path = data['LR_path'][0]
                img_name = os.path.splitext(os.path.basename(img_path))[0]

                sr_img = util.tensor2img(model.generate())

                suffix = self.config['suffix']
                if suffix:
                    save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = os.path.join(dataset_dir, img_name + '.png')
                util.save_img(sr_img, save_img_path)
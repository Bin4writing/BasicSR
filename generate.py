from config import Config 


import os
import util as util
from data import create_dataloader
from models.SRRaGAN_model import SRRaGANModel
from data.LRHR_dataset import LRHRDataset




class Generator():
    def __init__(self,path='config/generate.yaml'):
        self.config = Config(path).load(False).setMissingNone()
        self._model = None
        generate_conf = self.config['datasets']['generate']
        self.root = os.path.dirname(generate_conf['dataroot_LR'])
        self.result = self.config['path']['results_root']
        self._lr_dir = generate_conf['dataroot_LR']
        self._name = generate_conf['dataroot_LR'].split('/')[-1]

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
            self.log(dataset_opt)
            data_set = LRHRDataset(dataset_opt)
            data_loader = create_dataloader(data_set, dataset_opt)
            self.log('Number of images in [{:s}]: {:d}'.format(self._name, len(data_set)))
            data_loaders.append(data_loader)

        for data_loader in data_loaders:
            self.log('\nGenerating from [{:s}]...'.format(self._name))
            dataset_dir = os.path.join(self.result, self._name)
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
if __name__ == '__main__':
    generator = Generator()
    generator.run()

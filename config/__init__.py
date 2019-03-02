import os
import os.path as osp
from collections import OrderedDict
import json
import yaml
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

class NoneDict(dict):
    def __missing__(self, key):
        return None
    @staticmethod
    def fromDict(it):
        def __recur(it):
            if isinstance(it,dict):
                converted = NoneDict()
                for k,v in it.items():
                    converted.update({k: __recur(v)})
                return converted
            elif isinstance(it,list):
                return [ __recur(v) for v in it]
            else:
                return it
        if not isinstance(it,dict):
            return NoneDict()
        else:
            return __recur(it)
class Config():
    def __init__(self, path):
        self.path = path
        self._conf = {
            'root': os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        }
    def load(self,is_train=True,path=None):
        if path: self.path = path
        with open(self.path,'r',encoding='utf-8') as f:
            self._conf.update(ordered_load(f.read()))
        self._conf['is_train'] = is_train
        for phase, dataset in self._conf['datasets'].items():
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            is_lmdb = False
            if 'hr_dir' in dataset and dataset['hr_dir'] is not None:
                dataset['hr_dir'] = os.path.expanduser(dataset['hr_dir'])
                if dataset['hr_dir'].endswith('lmdb'):
                    is_lmdb = True
            if 'lr_dir' in dataset and dataset['lr_dir'] is not None:
                dataset['lr_dir'] = os.path.expanduser(dataset['lr_dir'])
                if dataset['lr_dir'].endswith('lmdb'):
                    is_lmdb = True
            dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        if is_train:
            experiments_root = os.path.join(self._conf['root'], 'experiments', self._conf['name'])
            self._conf['experiments_root'] = experiments_root
            self._conf['models'] = os.path.join(experiments_root, 'models')
            self._conf['training_state'] = os.path.join(experiments_root, 'training_state')
            self._conf['log'] = experiments_root
            self._conf['val_images'] = os.path.join(experiments_root, 'val_images')
        else:
            result_dir = os.path.join(self._conf['root'], 'results', self._conf['name'])
            self._conf['result_dir'] = result_dir
            self._conf['log'] = result_dir

        gpu_list = ','.join(str(x) for x in self._conf['GPUs'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        return self

    def __getitem__(self,key):
        return self._conf[key]

    def __setitem__(self,key,value):
        self._conf[key] = value

    def __str__(self):
        def __recur(it,indent=1):
            msg = ''
            for k, v in it.items():
                if isinstance(v, dict):
                    msg += ' ' * (indent * 2) + k + ':[\n'
                    msg += __recur(v, indent + 1)
                    msg += ' ' * (indent * 2) + ']\n'
                else:
                    msg += ' ' * (indent * 2) + k + ': ' + str(v) + '\n'
            return msg
        return __recur(self._conf)

    def setMissingNone(self):
        self._conf = NoneDict.fromDict(self._conf)
        return self

    def checkResume(self):
        if self._conf['resume_state']:
            state_idx = osp.basename(self._conf['resume_state']).split('.')[0]
            self._conf['GAN']['path'] = osp.join(self._conf['models'],
                                                    '{}_G.pth'.format(state_idx))
            if 'gan' in self._conf['model']:
                self._conf['Discriminator']['path'] = osp.join(self._conf['models'],
                                                        '{}_D.pth'.format(state_idx))
        return self

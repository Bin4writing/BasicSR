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
        self._conf = {}
        ext = os.path.splitext(path)[-1]
        with open(self.path,'r',encoding='utf-8') as f:
            self._conf.update(ordered_load(f.read()))
			
    def load(self,is_train=True):
        scale = self._conf['scale']
        self._conf['is_train'] = is_train
        for phase, dataset in self._conf['datasets'].items():
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            dataset['scale'] = scale
            is_lmdb = False
            if 'dataroot_HR' in dataset and dataset['dataroot_HR'] is not None:
                dataset['dataroot_HR'] = os.path.expanduser(dataset['dataroot_HR'])
                if dataset['dataroot_HR'].endswith('lmdb'):
                    is_lmdb = True
            if 'dataroot_LR' in dataset and dataset['dataroot_LR'] is not None:
                dataset['dataroot_LR'] = os.path.expanduser(dataset['dataroot_LR'])
                if dataset['dataroot_LR'].endswith('lmdb'):
                    is_lmdb = True
            dataset['data_type'] = 'lmdb' if is_lmdb else 'img'

        for key, path in self._conf['path'].items():
            if path and key in self._conf['path']:
                self._conf['path'][key] = os.path.expanduser(path)
        if is_train:
            experiments_root = os.path.join(self._conf['path']['root'], 'experiments', self._conf['name'])
            self._conf['path']['experiments_root'] = experiments_root
            self._conf['path']['models'] = os.path.join(experiments_root, 'models')
            self._conf['path']['training_state'] = os.path.join(experiments_root, 'training_state')
            self._conf['path']['log'] = experiments_root
            self._conf['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        else:
            results_root = os.path.join(self._conf['path']['root'], 'results', self._conf['name'])
            self._conf['path']['results_root'] = results_root
            self._conf['path']['log'] = results_root


        self._conf['network_G']['scale'] = scale

        gpu_list = ','.join(str(x) for x in self._conf['gpu_ids'])
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
        if self._conf['path']['resume_state']:
            if self._conf['path']['pretrain_model_G'] or self._conf['path']['pretrain_model_D']:
                print('resume training')

            state_idx = osp.basename(self._conf['path']['resume_state']).split('.')[0]
            self._conf['path']['pretrain_model_G'] = osp.join(self._conf['path']['models'],
                                                    '{}_G.pth'.format(state_idx))
            print('Set [pretrain_model_G] to ' + self._conf['path']['pretrain_model_G'])
            if 'gan' in self._conf['model']:
                self._conf['path']['pretrain_model_D'] = osp.join(self._conf['path']['models'],
                                                        '{}_D.pth'.format(state_idx))
                print('Set [pretrain_model_D] to ' + self._conf['path']['pretrain_model_D'])
        return self
    


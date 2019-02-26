import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict
from config import Config
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
opt = Config('options/test/test_ESRGAN.yaml').load().setMissingNone()

util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(opt)
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    logger.info(dataset_opt)
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        need_HR = False

        model.feed_data(data, need_HR=need_HR)
        img_path = data['LR_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        model.test()  # test
        visuals = model.get_current_visuals(need_HR=need_HR)

        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = os.path.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        

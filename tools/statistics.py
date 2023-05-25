import pickle
import time
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
from pcdet.models import load_data_to_gpu

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils
from pcdet.utils.statistics_utils import statistics


class Recoder:
    def __init__(self, model, dataloader, save_path, ckpt_dir_path):
        self.model = model
        self.dataloader = dataloader
        self.save_path = save_path

    def forward(self):
        # load checkpoint
        for i, batch_dict in enumerate(self.dataloader):
            load_data_to_gpu(batch_dict)
            pred_dicts, _ = self.model(batch_dict)
            self.recode_results(pred_dicts, batch_dict)
    
    def make_ckpt_array(self):
        pass
    
    def recode_results(self, pred_dicts, batch_dict):
        pass


class Analyzer:
    def __init__(self, root_path):
        self.save_path = root_path
    
    def Analyze(self, method):
        method_func = getattr(self, method)
        method_func()
        pass

    def analyze_recall_precision(self):
        pass


#main
if __name__ == '__main__':
    import sys
    assert sys.argv.__len__() > 1, 'There is no input.'
    if sys.argv[1] == 'recode':
        assert sys.argv.__len__() == 4, 'model_cfg, ckpt_dir, save_path'
        model_cfg = sys.argv[2]
        cfg_from_yaml_file(model_cfg, cfg)

        ckpt_dir = sys.argv[3]
        ckpt_dir = Path(ckpt_dir)
                
        save_path = sys.argv[4]
        save_path = Path(save_path)
        
        log_file = save_path / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # build dataloader
        if cfg.get('TEST_DATA_CONFIG', None) is not None:
            cfg.DATA_CONFIG = cfg.TEST_DATA_CONFIG

        test_set, dataloader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=sys.argv.batch_size,
            dist=sys.argv, workers=sys.argv.workers, logger=logger, training=False
        )

        # build network
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        
        # recoder = Recoder(model, dataloader, save_path, ckpt_dir)
        recoder = Recoder(model, dataloader, save_path, ckpt_dir)

    elif sys.argv[1] == 'analyze':
        assert sys.argv.__len__() == 5, 'model_cfg, ckpt_dir, root_path, method'
        # build analyzer
        # analyzer = Analyzer(save_path)



# def RecordStatistics(self, pred_dicts, dataloader, draw_scenes=False, **kwargs):
#     for i, batch_dict in enumerate(dataloader):
#         load_data_to_gpu(batch_dict)

#         if draw_scenes:
#             from visual_utils.open3d_vis_utils import draw_batch_scenes
#             draw_batch_scenes(batch_dict, pred_dicts)
        
#         for i in range(batch_dict['batch_size']):
#             self.sta_dicts = statistics(pred_dicts[i]['pred_boxes'], pred_dicts[i]['pred_labels'],
#                           batch_dict['gt_boxes'][i][:,:7], batch_dict['gt_boxes'][i][:,7])
#             self.sta_dicts['tp'] += self.sta_dicts['tp']
#             self.sta_dicts['fp'] += self.sta_dicts['fp']
#             self.sta_dicts['fn'] += self.sta_dicts['fn']

#             self.sta_dicts['iou_scores'] = pred_dicts[i]['pred_scores']
#             self.sta_dicts['cls_scores'] = pred_dicts[i]['pred_labels']
#             self.sta_dicts['gt_iou_scores'] = batch_dict['gt_boxes'][i][:,7]
#             self.sta_dicts['gt_cls_scores'] = batch_dict['gt_boxes'][i][:,7]
#             self.sta_dicts['precision'] = self.sta_dicts['tp'] / (self.sta_dicts['tp'] + self.sta_dicts['fp'])
#             self.sta_dicts['recall'] = self.sta_dicts['tp'] / (self.sta_dicts['tp'] + self.sta_dicts['fn'])
#             self.sta_dicts['f1_score'] = 2 * self.sta_dicts['precision'] * self.sta_dicts['recall'] / (self.sta_dicts['precision'] + self.sta_dicts['recall'])
#             self.sta_dicts['ap'] = np.mean(self.sta_dicts['precision'])
#             self.sta_dicts['ar'] = np.mean(self.sta_dicts['recall'])

# def drawing_IoUGraph():


#     return

# def drawing_ClassGraph():

#     return

# def SaveStatistics(self, statistics_dir=None):
#     """
#     the directory to save statistics.pkl
#     Args:
#         sta_dict:
#             bbox: bounding box
#             gt_bbox: ground truth bounding box
#             tp: true positive
#             fp: false positive
#             fn: false negative
#             iou_scores: IoU scores
#             cls_scores: Class scores
#             gt_iou_scores: ground truth IoU
#             gt_cls: ground truth Class
#         statistics_dir: the directory to save statistics.pkl
#     Returns:

#     """

#     if statistics_dir is None:
#         statistics_dir = './statistics'
#     statistics_dir = Path(statistics_dir)
#     if not statistics_dir.exists():
#         statistics_dir.mkdir()
#     statistics_pkl = 'statistics_%s.pkl' % time.strftime("%m-%d_%H-%M-%S", time.localtime())

#     with open(statistics_dir / statistics_pkl, 'wb') as f:
#         pickle.dump(self.sta_dicts, f)
    


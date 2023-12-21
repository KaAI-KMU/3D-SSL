import copy
import numpy as np
import torch
import os
import kornia

from collections import OrderedDict

from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from ...utils.ssl_utils import *


class PVRCNNSSL(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        model_cfg_copy = copy.deepcopy(model_cfg)
        dataset_copy = copy.deepcopy(dataset)

        self.filtering = model_cfg.SSL_CONFIG.get('FILTERING_CONFIG', None) is not None
        if self.filtering:
            cls_threshold = np.array(model_cfg.SSL_CONFIG.FILTERING_CONFIG.CLS_THRESHOLD)
            iou_threshold = np.array(model_cfg.SSL_CONFIG.FILTERING_CONFIG.IOU_THRESHOLD)
            self.thresholds = np.stack((cls_threshold, iou_threshold))

        self.pv_rcnn = PVRCNN(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.pv_rcnn_ema = PVRCNN(model_cfg=model_cfg_copy, num_class=num_class, dataset=dataset_copy)
        for param in self.pv_rcnn_ema.parameters():
            param.detach_()
        
        self.add_module('pv_rcnn', self.pv_rcnn)
        self.add_module('pv_rcnn_ema', self.pv_rcnn_ema)

        self.global_step_ema = 0

    def forward(self, batch_dict):
        if self.training:
            batch_size = batch_dict['batch_size']
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                self.pv_rcnn_ema.eval()
                for cur_module in self.pv_rcnn_ema.module_list:
                    batch_dict = cur_module(batch_dict)
                pred_dicts, _ = self.pv_rcnn_ema.post_processing(batch_dict)

            # Generate batch_dict with pseudo labels
            pseudo_label = copy.deepcopy(pred_dicts)
            for batch_idx in range(batch_size):
                if self.filtering:
                    scores = torch.stack((pred_dicts[batch_idx]['pred_cls_scores'], pred_dicts[batch_idx]['pred_scores']))
                    selected = box_filtering_by_score(pred_dicts[batch_idx]['pred_boxes'], pred_dicts[batch_idx]['pred_labels'], scores, self.thresholds)
                    for key, val in pred_dicts[batch_idx].items():
                        pseudo_label[batch_idx][key] = val[selected]
                else:
                    pseudo_label = pred_dicts
            batch_dict_train = self.dataset.generate_datadict(pseudo_label, batch_dict)
            del batch_dict

            # Visualize pseudo labels for debugging
            if False:
                from visual_utils.open3d_vis_utils import draw_batch_scenes
                draw_batch_scenes(batch_dict_train)

            # Train PV-RCNN with pseudo labels
            load_data_to_gpu(batch_dict_train)
            self.pv_rcnn.train()
            for cur_module in self.pv_rcnn.module_list:
                batch_dict_train = cur_module(batch_dict_train)

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            for cur_module in self.pv_rcnn.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.pv_rcnn.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.pv_rcnn.dense_head.get_loss()
        loss_point, tb_dict = self.pv_rcnn.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.pv_rcnn.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_point + loss_rcnn
        
        return loss, tb_dict, disp_dict
    
    @torch.no_grad()
    def update_global_step(self):
        self.global_step_ema += 1
        ema_keep_rate = 0.9996
        change_global_step = 2000
        if self.global_step_ema < change_global_step:
            keep_rate = (ema_keep_rate - 0.5) / change_global_step * self.global_step_ema + 0.5
        else:
            keep_rate = ema_keep_rate

        student_model_dict = self.pv_rcnn.state_dict()
        new_teacher_dict = OrderedDict()
        for key, value in self.pv_rcnn_ema.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                )
            else:
                raise NotImplementedError
        self.pv_rcnn_ema.load_state_dict(new_teacher_dict)

    def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            # load pretrain model
            new_key = 'pv_rcnn.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

            new_key = 'pv_rcnn_ema.' + key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

            new_key = key
            if new_key in self.state_dict() and self.state_dict()[new_key].shape == model_state_disk[key].shape:
                update_model_state[new_key] = val

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'score_keys']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
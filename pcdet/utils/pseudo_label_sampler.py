import torch
import numpy as np
import pickle
import copy
from tqdm import tqdm

from pathlib import Path

from .ssl_utils import box_filtering_by_score
from ..ops.roiaware_pool3d import roiaware_pool3d_utils
from ..models import load_data_to_gpu
from ..datasets import build_dataloader


class PseudoLabelSampler:
    def __init__(self, sampler_cfg, model, dataloader):
        self.sampler_cfg = sampler_cfg
        self.interval = sampler_cfg['INTERVAL']
        self.model = model
        self.root_path = dataloader.dataset.root_path
        self.class_names = dataloader.dataset.class_names

        split_name = sampler_cfg['SPLIT_NAME']
        self.database_save_path = Path(self.root_path) / ('gt_database_runtime_%s' % split_name)
        self.db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_runtime_%s.pkl' % split_name)

        imageset_file = self.root_path / 'ImageSets' / ('train_%s.txt' % split_name)
        self.labeled_mask = np.loadtxt(imageset_file, dtype=np.int32)

        _, self.dataloader, _ = build_dataloader(
            dataset_cfg=copy.deepcopy(dataloader.dataset.dataset_cfg), class_names=self.class_names,
            dist=False, batch_size=6, logger=None, training=False
        )
        self.class_names = np.array(self.class_names)

    def clear_database(self):
        import shutil
        if self.database_save_path.exists():
            shutil.rmtree(str(self.database_save_path))
        self.database_save_path.mkdir(parents=False, exist_ok=False)
        if self.db_info_save_path.exists():
            self.db_info_save_path.unlink()

    def sample_pseudo_labels(self):
        self.clear_database()

        self.model.eval()
        all_db_infos = {}
        for batch_dict in tqdm(self.dataloader, desc='pseudo_label_sampling',leave=True):
            batch_size = batch_dict['batch_size']
            labeled_indices = [int(batch_dict['frame_id'][batch_idx]) in self.labeled_mask for batch_idx in range(batch_size)]
            load_data_to_gpu(batch_dict)
            with torch.no_grad():
                pred_dicts, _ = self.model(batch_dict)

            # Filter boxes by score
            cls_threshold = np.array(self.sampler_cfg.CLS_THRESHOLD)
            iou_threshold = np.array(self.sampler_cfg.IOU_THRESHOLD)
            thresholds = np.stack((cls_threshold, iou_threshold))

            # Generate batch_dict with pseudo labels
            pseudo_label = copy.deepcopy(pred_dicts)
            for batch_idx in range(batch_size):
                scores = torch.stack((pred_dicts[batch_idx]['pred_cls_scores'], pred_dicts[batch_idx]['pred_scores']))
                selected = box_filtering_by_score(pred_dicts[batch_idx]['pred_boxes'], pred_dicts[batch_idx]['pred_labels'], scores, thresholds)
                for key, val in pred_dicts[batch_idx].items():
                    pseudo_label[batch_idx][key] = val[selected]
            del pred_dicts

            pseudo_label_dict = self.generate_single_db(pseudo_label, batch_dict, labeled_indices, all_db_infos)
        self.save_db_infos(pseudo_label_dict)

    def generate_single_db(self, pseudo_labels, batch_dict, labeled_mask, db_infos):
        batch_size = batch_dict['batch_size']
        for batch_idx in range(batch_size):
            pseudo_boxes = pseudo_labels[batch_idx]['pred_boxes'].cpu().detach().numpy()
            num_obj = pseudo_boxes.shape[0]
            if labeled_mask[batch_idx] or num_obj==0:
                continue

            sample_idx = batch_dict['frame_id'][batch_idx]
            points_indices = batch_dict['points'][:,0] == batch_idx
            points = batch_dict['points'][points_indices][:,1:].cpu().detach().numpy()
            pseudo_names = np.array(self.class_names[pseudo_labels[batch_idx]['pred_labels'].cpu().detach().numpy()-1])

            iou_scores = pseudo_labels[batch_idx]['pred_scores'].cpu().detach().numpy()
            cls_scores = pseudo_labels[batch_idx]['pred_cls_scores'].cpu().detach().numpy()
            bbox = np.zeros([num_obj, 4])
            difficulty = np.zeros_like(pseudo_names, dtype=np.int32)

            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(pseudo_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, pseudo_names[i], i)
                filepath = self.database_save_path / filename
                if filepath.exists():
                    continue

                pseudo_points = points[point_indices[i] > 0]
                pseudo_points[:, :3] -= pseudo_boxes[i, :3]
                with open(filepath, 'w') as f:
                    pseudo_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))
                db_info = {'name': pseudo_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                            'box3d_lidar': pseudo_boxes[i], 'num_points_in_gt': pseudo_points.shape[0],
                            'difficulty': difficulty[i], 'bbox': bbox[i], 'score': -1.0,
                            'iou_score': iou_scores[i], 'cls_score': cls_scores[i]}
                if pseudo_names[i] in db_infos:
                    db_infos[pseudo_names[i]].append(db_info)
                else:
                    db_infos[pseudo_names[i]] = [db_info]
        return db_infos

    def save_db_infos(self, db_infos):
        with open(self.db_info_save_path, 'wb') as f:
            pickle.dump(db_infos, f)
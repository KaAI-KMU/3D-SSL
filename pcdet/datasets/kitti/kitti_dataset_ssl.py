import copy
import pickle

import numpy as np
import torch
from skimage import io

from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.kitti import kitti_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti, ssl_utils
from pcdet.datasets.dataset import DatasetTemplate


class KittiDatasetSSL(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        split_name = dataset_cfg.get('SPLIT_NAME', None)
        dataset_cfg['DATA_AUGMENTOR']['SPLIT_NAME'] = split_name
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        if split_name is not None:
            imageset_file = self.root_path / 'ImageSets' / ('train_%s.txt' % split_name)
            self.label_indices = np.loadtxt(imageset_file, dtype=np.int32)
            self.class_names_map = {i+1: class_name for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index, pseudo_label_dict=None):
        if not self.training: # for evaluation
            return super().__getitem__(index)
        
        data_dict = super().__getitem__(index, prepare_data=False)
        # generate unlabeled data with pseudo labels
        if pseudo_label_dict is not None:
            data_dict.update({
                'gt_names': pseudo_label_dict['gt_names'],
                'gt_boxes': pseudo_label_dict['gt_boxes'],
            })
            for key in self.score_keys:
                data_dict[key] = pseudo_label_dict[key]
            data_dict['disable_gt_sampling'] = False
            data_dict = self.prepare_data(data_dict, no_aug=False, no_regen=True)
        # data for pseudo labeling
        else:
            for key in self.score_keys:
                data_dict[key] = np.ones((data_dict['gt_boxes'].shape[0],), dtype=np.float32)
            data_dict = self.prepare_data(data_dict, no_aug=True, no_regen=False)

        data_dict['image_shape'] = self.kitti_infos[index]['image']['image_shape']
        data_dict['data_index'] = index
        return data_dict

    def generate_datadict(self, pred_dicts, batch_dict):
        batch_size = batch_dict['batch_size']
        batch_frame_id = batch_dict['frame_id']
        labeled_idx_mask = [int(frame) in self.label_indices for frame in batch_frame_id]

        # Convert tensors to numpy arrays
        for batch_idx in range(batch_size):
            for key, val in pred_dicts[batch_idx].items():
                pred_dicts[batch_idx][key] = val.cpu().detach().numpy()

        data_list = []
        for batch_idx in range(batch_size):
            data_index = int(batch_dict['data_index'][batch_idx])
            pseudo_label_dict = None
            if not labeled_idx_mask[batch_idx]: # generate unlabeled data for training
                pseudo_label_names = np.array([self.class_names_map[i] for i in pred_dicts[batch_idx]['pred_labels']])
                pseudo_label_dict = {
                    'gt_names': pseudo_label_names,
                    'gt_boxes': pred_dicts[batch_idx]['pred_boxes'],
                    'iou_scores': pred_dicts[batch_idx]['pred_scores'],
                    'cls_scores': pred_dicts[batch_idx]['pred_cls_scores'],
                }
                data_dict = self.__getitem__(data_index, pseudo_label_dict=pseudo_label_dict)
            else: # generate labeled data for training
                data_dict = super().__getitem__(data_index)
            data_list.append(data_dict)
        
        batch_dict = self.collate_batch(data_list)
        return batch_dict
    
    def create_db_infos(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s_' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s_%s.pkl' % (split, self.dataset_cfg['SPLIT_NAME']))

        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        cnt = 0
        for k in range(len(infos)):
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            if int(sample_idx) not in self.label_indices:
                continue

            cnt += 1
            print('gt_database sample: %d/%d' % (cnt, len(self.label_indices)))
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i],
                               'iou_scores': 1.0, 'cls_scores': 1.0}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)


def create_split_db_infos(dataset_cfg, class_names, data_path, save_path):
    dataset = KittiDatasetSSL(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split = 'train'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_db_infos(train_filename, split=train_split)
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_split_db_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg['SPLIT_NAME'] = sys.argv[3]
        create_split_db_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )    
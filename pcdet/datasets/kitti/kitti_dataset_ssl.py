import pickle
import numpy as np

from pcdet.datasets.kitti.kitti_dataset import KittiDataset


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
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, pretraining=False, root_path=root_path, logger=logger
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
    
    def __len__(self):
        length = len(self.kitti_infos) * self.repeat if self.training else len(self.kitti_infos)

        if self._merge_all_iters_to_one_epoch:
            return length * self.total_epochs

        return length

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
    

def create_split_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDatasetSSL(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split = 'train'
    labeled_split_name = dataset_cfg['SPLIT_NAME']
    train_filename = save_path / ('kitti_infos_%s_%s.pkl' % (train_split, labeled_split_name))

    print('---------------Start to generate data infos---------------')
    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split, labeled_split_name=labeled_split_name)
    print('---------------Data preparation Done---------------')

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_split_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg['SPLIT_NAME'] = sys.argv[3]
        create_split_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )    
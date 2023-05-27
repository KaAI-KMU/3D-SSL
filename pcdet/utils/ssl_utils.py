import torch
import numpy as np

from pathlib import Path

from . import common_utils
from ..ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


class TemporalEnsemble:
    def __init__(self, output_dir, te_config):
        self.pseudo_label_dir = output_dir / 'pseudo_labels'
        self.pseudo_label_dir.mkdir(exist_ok=True)

        self.matching_thresh = te_config.MATCHING_THRESH
        self.alpha = te_config.ALPHA
        self.unmatched_count_thresh = te_config.UNMATCHED_COUNT_THRESH
        self.heading_thresh = te_config.DISABLE_AVERAGE_HEADING_THRESH

    
    def get_pre_pseudo_label(self, frame_id):
        pseudo_label_path = self.pseudo_label_dir / (str(frame_id)+'.npy')
        pseudo_label = None
        if pseudo_label_path.exists():
            pseudo_label = np.load(pseudo_label_path)
            pseudo_label = torch.from_numpy(pseudo_label).cuda()

        return pseudo_label
    

    def save_new_pseudo_label(self, pseudo_label, frame_id):
        pseudo_label_path = self.pseudo_label_dir / str(frame_id)
        np.save(pseudo_label_path, pseudo_label.cpu().numpy())


    def step(self, frame_ids, pred_dicts, batch_dict):
        batch_size = len(pred_dicts)
        pseudo_label_dicts = []
        for i in range(batch_size):
            pseudo_label = self.get_pre_pseudo_label(frame_ids[i])
            new_pseudo_label = torch.cat((pred_dicts[i]['pred_boxes'], pred_dicts[i]['pred_scores'].view(-1,1), pred_dicts[i]['pred_cls_scores'].view(-1,1),\
                                          pred_dicts[i]['pred_labels'].view(-1,1), torch.zeros((pred_dicts[i]['pred_boxes'].shape[0], 2), dtype=torch.float32).cuda()), dim=1)

            if pseudo_label is not None:
                matched_indices_old, matched_indices_new, unmatched_indices, new_indices = self.match_boxes(pseudo_label[:,:7], new_pseudo_label[:,:7], self.matching_thresh)

                cls_changed_idx = torch.where(pseudo_label[matched_indices_old,9] != new_pseudo_label[matched_indices_new,9])[0]
                cls_changed_idx_old = matched_indices_old[cls_changed_idx.cpu().numpy()]
                cls_changed_idx_new = matched_indices_new[cls_changed_idx.cpu().numpy()]
                cls_changed_boxes = torch.where((pseudo_label[cls_changed_idx_old,8] < new_pseudo_label[cls_changed_idx_new,8]).view(-1,1), \
                                                new_pseudo_label[cls_changed_idx_new], pseudo_label[cls_changed_idx_old])

                pseudo_label[matched_indices_old,-2] += 1
                pseudo_label[unmatched_indices,-2:] += 1

                alpha = torch.ones((len(matched_indices_old), 1)).cuda() * self.alpha
                alpha = torch.cat((1 - 1 / (pseudo_label[matched_indices_old,-2].view(-1,1) + 1), alpha), dim=1).min(dim=1).values.view(-1,1)
                updated_pseudo_label = pseudo_label.detach().clone()
                updated_pseudo_label[matched_indices_old,:9] = alpha * pseudo_label[matched_indices_old,:9] + (1 - alpha) * new_pseudo_label[matched_indices_new, :9]
                
                disable_heading_average_indices = (abs(pseudo_label[matched_indices_old,6] - new_pseudo_label[matched_indices_new,6]) > self.heading_thresh).cpu().numpy()
                if disable_heading_average_indices.any():
                    heading_update_old = matched_indices_old[disable_heading_average_indices]
                    heading_update_new = matched_indices_new[disable_heading_average_indices]   
                    updated_heading = torch.where(new_pseudo_label[heading_update_new,7] > pseudo_label[heading_update_old,7],\
                                                        new_pseudo_label[heading_update_new,6], pseudo_label[heading_update_old,6])
                    updated_pseudo_label[heading_update_old,6] = updated_heading

                if len(cls_changed_idx) != 0:
                    updated_pseudo_label[cls_changed_idx] = cls_changed_boxes

                updated_pseudo_label = updated_pseudo_label[updated_pseudo_label[:,-1] < self.unmatched_count_thresh]                
                updated_pseudo_label = torch.cat((updated_pseudo_label, new_pseudo_label[new_indices]),dim=0)
                if False:
                    from visual_utils import open3d_vis_utils as V
                    points = batch_dict['points']
                    points_idx = points[:,0] == i
                    V.draw_scenes(points=points[points_idx,1:], ref_boxes=pseudo_label, gt_boxes=new_pseudo_label, additional_boxes=updated_pseudo_label)

            else:
                updated_pseudo_label = new_pseudo_label

            self.save_new_pseudo_label(pseudo_label=updated_pseudo_label, frame_id=frame_ids[i])
            pseudo_label_dicts.append({'pred_boxes':updated_pseudo_label[:,:7], 'pred_scores':updated_pseudo_label[:,7], \
                                       'pred_cls_scores':updated_pseudo_label[:,8], 'pred_labels':updated_pseudo_label[:,9]})
        
        return pseudo_label_dicts
    

    @staticmethod
    def match_boxes(boxes_a, boxes_b, thresh=0.1):
        '''
        Args :
            boxes : [[x, y, z, l, w, h, theta]]
            boxes_a : M
            boxes_b : N
            thresh : matching threshold 
        Returns :
            matched_indices_a, unmatched_indices_a : sorted by ascending
            matched_indices_b, unmatched_indices_b : ordered by matching with matched_indices_a
        '''
        boxes_a, _ = common_utils.check_numpy_to_torch(boxes_a)
        boxes_b, _ = common_utils.check_numpy_to_torch(boxes_b)

        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        iou = boxes_iou3d_gpu(boxes_a, boxes_b) # (M, N)

        matched_indices_a = []
        matched_indices_b = []
        unmatched_indices_a = []
        max_iou_values_a, max_iou_indices_a = iou.max(dim=1)
        max_iou_indices_a[max_iou_values_a < thresh] = -1
        max_iou_values_b, max_iou_indices_b = iou.max(dim=0)
        max_iou_indices_b[max_iou_values_b < thresh] = -1

        for i in range(len(max_iou_indices_a)):
            if max_iou_indices_a[i] != -1 and max_iou_indices_b[max_iou_indices_a[i]] == i:
                matched_indices_a.append(i)
                matched_indices_b.append(int(max_iou_indices_a[i]))
            else:
                unmatched_indices_a.append(i)
        unmatched_indices_b = [i for i in range(boxes_b.shape[0]) if i not in matched_indices_b]

        return np.array(matched_indices_a), np.array(matched_indices_b), np.array(unmatched_indices_a), np.array(unmatched_indices_b)


def box_filtering_by_score(boxes, classes, scores, threshold_list):
    '''
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, theta]
        classes: (N) int
        scores: (N) float
        threshold_list: (num_threshold_type, num_class) float
    Returns:
        selected: (N) bool
    '''
    if boxes.numel() == 0:
        return torch.tensor([]).cuda().long()

    threshold_list, _ = common_utils.check_numpy_to_torch(threshold_list)
    scores, _ = common_utils.check_numpy_to_torch(scores)

    if len(threshold_list.shape) == 1:
        threshold_list = threshold_list.reshape([1, -1])
    if len(scores.shape) == 1:
        scores = scores.reshape([1, -1])
    num_threshold_type = threshold_list.shape[0]

    assert int(max(classes)) <= len(threshold_list[0])
    assert scores.shape[0] == num_threshold_type
    assert len(boxes.shape) == 2

    num_boxes = boxes.shape[0]
    thresholds = torch.transpose(threshold_list.cuda().unsqueeze(0).repeat(num_boxes, 1, 1).gather(dim=2, index=classes.view(-1,1,1).repeat(1, num_threshold_type, 1)-1).squeeze(-1), 1, 0)
    selected = (scores > thresholds).all(dim=0)

    return selected


# only available with 1 batch. used only for debugging
def augment_boxes(batch):
    for batch_idx in range(batch['batch_size']):
        boxes = batch['src_gt_boxes'][batch_idx]
        if batch['flip_x'][batch_idx]:
            boxes[:, 1] = -boxes[:, 1]
            boxes[:, 6] = -boxes[:, 6]

            if boxes.shape[1] > 7:
                raise NotImplementedError
            
        noise_rotation = batch['noise_rot'][batch_idx]
        boxes[:, 0:3] = common_utils.rotate_points_along_z(boxes.cpu()[np.newaxis, :, 0:3], np.array([noise_rotation.cpu()]))
        boxes[:, 6] += noise_rotation

        noise_scale = batch['noise_scale'][batch_idx]
        boxes[:, :6] *= noise_scale

        batch['src_gt_boxes'] = boxes

    return batch

def cluster_boxes(boxes, classes, iou_thresh, class_scores):
    '''
    if clusters in boxes have different classes, the cluster will be assigned to the class with the highest score
    and the cluster indices of the boxes which have lower class score will be assigned to -1    
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, theta]
        classes: (N) int
        iou_thresh: float
        class_scores: (N) float
    Returns:
        cluster_indices: (N) int
    '''
    if boxes.numel() == 0:
        return torch.tensor([]).cuda().long()
    
    boxes, _ = common_utils.check_numpy_to_torch(boxes)
    classes, _ = common_utils.check_numpy_to_torch(classes)
    class_scores, _ = common_utils.check_numpy_to_torch(class_scores)

    assert len(boxes.shape) == 2
    assert len(classes.shape) == 1
    assert len(class_scores.shape) == 1
    assert boxes.shape[0] == classes.shape[0]
    assert boxes.shape[0] == class_scores.shape[0]
    assert boxes.shape[1] == 7

    iou = boxes_iou3d_gpu(boxes, boxes) # (N, N)
    iou = iou - torch.eye(iou.shape[0]).cuda() # (N, N)
    iou[iou < iou_thresh] = 0
    iou[iou >= iou_thresh] = 1

    cluster_indices = torch.zeros_like(classes).long()
    cluster_idx = 0
    for i in range(boxes.shape[0]):
        if cluster_indices[i] != 0:
            continue
        cluster_idx += 1
        cluster_indices[i] = cluster_idx
        cluster_indices[iou[i] == 1] = cluster_idx
    cluster_indices = cluster_indices.cuda()

    # assign cluster to the class with the highest score and the boxes which have lower class score will be assigned to -(cluster_index)
    for i in range(1,cluster_idx+1):
        cluster_class_scores = class_scores[cluster_indices == i]
        cluster_class = classes[torch.argmax(cluster_class_scores)]
        cluster_indices[cluster_indices == i] = -i
        cluster_indices[(cluster_indices == -i) * (classes == cluster_class)] = i

    return cluster_indices
    

def plot_boxes_score(boxes, scores):
    '''
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, theata]
        scores: (N) float
    x-axis : boxes parameters
    y-axis : scores in each cluster
    '''
    import matplotlib.pyplot as plt

    if boxes.shape[0] == 0:
        return

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()

    params = ['x', 'y', 'z', 'l', 'w', 'h', 'theta']
    for i in range(7):
        ax1 = plt.subplot(2, 4, i+1)
        ax1.set_title(params[i])
        ax1.scatter(boxes[:, i], scores, s=0.3)

        mean = (boxes[:,i] * scores).sum() / scores.sum()
        mean_sq = (boxes[:,i]**2 * scores).sum() / scores.sum()
        var = mean_sq - mean**2
        std = np.sqrt(var)
        def gaussian(x):
            return np.exp(-(x - mean)**2 / (2 * std**2)) / (std * np.sqrt(2 * np.pi))
        x = np.linspace(boxes[:,i].min(), boxes[:,i].max(), 100)
        ax1.plot(x, gaussian(x), color='red', linewidth=0.5)

    plt.show()


def circle_func(x, tau):
    # y = -sqrt(tau^2 - x^2) + tau (x < tau)
    # y = sqrt((1-tau)^2 - (x-1)^2) + tau (tau <= x)
    # torch
    y = torch.zeros_like(x)
    y[x < tau] = -torch.sqrt(tau**2 - x[x < tau]**2) + tau
    y[x >= tau] = torch.sqrt((1-tau)**2 - (x[x >= tau]-1)**2) + tau
    return y

def ramp_func(x, tau):
    return x

weight_functions = {
    'circle_func': circle_func,
    'ramp_func': ramp_func,
}

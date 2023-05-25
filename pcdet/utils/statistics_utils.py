import torch

import numpy as np

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu


def statistics(pred_boxes, pred_labels, gt_boxes, gt_labels, thresholds=np.array([0.7, 0.5, 0.5])):
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(gt_labels, torch.Tensor):
        gt_labels = gt_labels.cpu().numpy()

    num_classes = thresholds.shape[0]
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)

    for i in range(num_classes):
        pred_boxes_i = pred_boxes[pred_labels == i+1]
        gt_boxes_i = gt_boxes[gt_labels == i+1]
        iou = boxes_iou3d_gpu(torch.from_numpy(pred_boxes_i).cuda(), torch.from_numpy(gt_boxes_i).cuda()).cpu().numpy()
        tp[i] = np.sum(np.any(iou > thresholds[i], axis=1))
        fp[i] = pred_boxes_i.shape[0] - tp[i]
        fn[i] = gt_boxes_i.shape[0] - tp[i]
    
    sta_dicts = {}
    sta_dicts['tp'] = tp
    sta_dicts['fp'] = fp
    sta_dicts['fn'] = fn

    return sta_dicts
import numpy as np
from .imageUtils import get_bbox_sz


def compute_score_by_centroid(pred_bbox, gt_bbox, tot=[20, 20]):
    pred_c = bbox2centroid(pred_bbox)
    gt_c = bbox2centroid(gt_bbox)
    diffs = abs(pred_c[:,None] - gt_c)
    x1, x2 = np.nonzero((diffs < tot).all(2))
    precision = np.unique(x1).shape[0]/pred_bbox.shape[0]
    recall = np.unique(x2).shape[0]/gt_bbox.shape[0]
    return recall, precision



def bbox2centroid(bboxes):
    return np.column_stack(((bboxes[:, 0] + bboxes[:, 2])/2, (bboxes[:, 1] + bboxes[:, 3])/2))


def evaluate_set_by_centroid(model, dataset, threshold=0.5, use_gpu=True):
    model.score_thresh = threshold
    recall_list = []
    precision_list = []
    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        recall, precision = compute_score_by_centroid(pred_bbox[0], gt_bbox)
        recall_list.append(recall)
        precision_list.append(precision)
    return recall_list, precision_list


def evaluate_set_by_defect_size(model, dataset, threshold=0.5, use_gpu=True):
    min_sz = 10
    max_sz = 100
    num_bins = 10
    splits = np.linspace(min_sz, max_sz, num=num_bins)
    tot_p = tot_g = tp_p = tp_g = [0] * (len(splits) + 1)

    model.score_thresh = threshold

    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        gt_sz = get_bbox_sz(gt_bbox)
        pred_sz = get_bbox_sz(pred_bbox)
        recall_index, precision_index = compute_score_detail_by_centroid(pred_bbox[0], gt_bbox)
        gt_tp_sz = gt_sz[recall_index]
        pred_tp_sz = pred_sz[precision_index]

        for i, split in enumerate(splits):
            tp_g[i] += np.sum(gt_tp_sz < split)
            tp_p[i] += np.sum(pred_tp_sz < split)
            tot_g[i] += np.sum(gt_sz < split)
            tot_p[i] += np.sum(pred_sz < split)
            if i == (splits.shape[0] - 1):
                tp_g[i+1] += np.sum(gt_tp_sz >= split)
                tp_p[i+1] += np.sum(pred_tp_sz >= split)
                tot_g[i+1] += np.sum(gt_sz >= split)
                tot_p[i+1] += np.sum(pred_sz >= split)

        recall_by_sz = [m / n for (m, n) in zip(tp_g, tot_g)]
        precision_by_sz = [m / n for (m, n) in zip(tp_p, tot_p)]
    return recall_by_sz, precision_by_sz


def compute_score_detail_by_centroid(pred_bbox, gt_bbox, tot=[20, 20]):
    pred_c = bbox2centroid(pred_bbox)
    gt_c = bbox2centroid(gt_bbox)
    diffs = abs(pred_c[:, None] - gt_c)
    x1, x2 = np.nonzero((diffs < tot).all(2))
    return x2, x1
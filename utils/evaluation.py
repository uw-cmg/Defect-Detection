import numpy as np


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


def evaluate_set_by_centroid(model, dataset, threshold = 0.6, use_gpu=True):
    model.score_thresh = 0.6
    recall_list = []
    precision_list = []
    for instance in dataset:
        img, gt_bbox, _ = instance
        pred_bbox, _, _ = model.predict([img])
        recall, precision = compute_score_by_centroid(pred_bbox, gt_bbox)
        recall_list.append(recall)
        precision_list.append(precision)
    return recall_list, precision_list



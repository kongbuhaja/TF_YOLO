import tensorflow as tf
from src.utils.util import make_anchors
from src.utils.metric import bbox_iou

class Sampler():
    def __init__(self, nc, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        self.nc = nc
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def sampling(self, pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        gt_shape = tf.shape(gt_labels)
        b, max_det = gt_shape[0], gt_shape[1]

        gt_labels = tf.cast(gt_labels, tf.int32)
        dtype = pred_scores.dtype

        mask_pos, metrics, overlaps = self.get_positive_sample(pred_scores,
                                                               pred_bboxes,
                                                               anc_points,
                                                               gt_labels,
                                                               gt_bboxes, 
                                                               mask_gt)
        target_gt_indices, fg_mask, mask_pos = self.get_highest_overlaps(mask_pos, overlaps, max_det)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_indices, fg_mask, dtype)

        metrics *= mask_pos
        pos_metrics = tf.reduce_max(metrics, -1, keepdims=True)
        pos_overlaps = tf.reduce_max(overlaps * mask_pos, -1, keepdims=True)
        norm_metrics = tf.reduce_max(metrics * pos_overlaps / (pos_metrics + self.eps), 1)[..., None]
        target_scores = target_scores * norm_metrics

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_indices
        # return anc_points, gt_bboxes, gt_labels, target_bboxes, target_labels, tf.cast(fg_mask, tf.bool)

    def get_positive_sample(self, pred_scores, pred_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        mask = self.get_mask_with_anchors(anc_points, gt_bboxes)

        metrics, overlaps = self.get_metrics(pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask * mask_gt)

        mask_topk = self.get_topk(metrics, topk_mask=mask_gt)
        mask_pos = mask_topk * mask * mask_gt

        return mask_pos, metrics, overlaps
    
    def get_highest_overlaps(self, mask_pos, overlaps, max_det):
        fg_mask = tf.reduce_sum(mask_pos, 1)
        conflict_mask = (fg_mask > 1)[:, None]

        max_overlaps_indices = tf.argmax(overlaps, axis=1, output_type=tf.int32)
        is_max_overlaps = tf.one_hot(max_overlaps_indices, depth=max_det, axis=1, dtype=mask_pos.dtype)
        
        mask_pos = tf.where(conflict_mask, is_max_overlaps, mask_pos)
        target_gt_indices = tf.argmax(mask_pos, axis=1, output_type=tf.int32)
        fg_mask = tf.cast(tf.reduce_sum(mask_pos, axis=1), tf.bool)
        
        return target_gt_indices, fg_mask, mask_pos
    
    def get_targets(self, gt_labels, gt_bboxes, target_gt_indices, fg_mask, dtype=tf.float32):
        gt_shape = tf.shape(gt_labels)
        b, max_det = gt_shape[0], gt_shape[1]
        batch_ind = tf.range(b, dtype=tf.int32)[:, None]
        target_gt_indices_flat = target_gt_indices + batch_ind * max_det
        
        gt_labels_flat = tf.reshape(gt_labels, [-1])
        target_labels = tf.gather(gt_labels_flat, target_gt_indices_flat)

        gt_bboxes_flat = tf.reshape(gt_bboxes, [-1, 4])
        target_bboxes = tf.gather(gt_bboxes_flat, target_gt_indices_flat)

        target_labels_clamped = tf.maximum(target_labels, 0)
        target_scores = tf.one_hot(tf.cast(target_labels_clamped, tf.int32), depth=self.nc, dtype=dtype)

        fg_scores_mask = tf.tile(fg_mask[:, :, None], [1, 1, self.nc])
        target_scores = tf.where(fg_scores_mask, target_scores, tf.cast(0.0, dtype))

        return target_labels, target_bboxes, target_scores

    def get_mask_with_anchors(self, anc_points, gt_bboxes):
        lt, rb = tf.split(gt_bboxes[:, :, None], 2, -1)
        deltas = tf.concat([anc_points[None, None] - lt, rb - anc_points[None, None]], -1)
        return tf.cast(tf.reduce_min(deltas, -1) > self.eps, anc_points.dtype)
    
    def get_metrics(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt):
        pred_scores_T = tf.transpose(pred_scores, [0, 2, 1])
        bbox_scores = tf.gather(pred_scores_T, gt_labels[..., 0], batch_dims=1)
        overlaps = tf.maximum(bbox_iou(gt_bboxes[:,:,None], pred_bboxes[:, None], xywh=False, CIoU=True)[..., 0], 0)
        bbox_scores = bbox_scores * mask_gt
        overlaps = overlaps * mask_gt

        metrics = tf.pow(bbox_scores, self.alpha) * tf.pow(overlaps, self.beta)

        return metrics, overlaps
    
    def get_topk(self, metrics, descending=True, topk_mask=None):
        shape = tf.shape(metrics)
        b, max_det, anc_n = shape[0], shape[1], shape[2]

        if not descending:
            metrics *= -1

        metric_flat = tf.reshape(metrics, [-1, anc_n])
        topk_values, topk_indices = tf.math.top_k(metric_flat, k=self.topk, sorted=True)

        if topk_mask is None:
            topk_mask = tf.cast(topk_values > self.eps, metrics.dtype)
        else:
            topk_mask = tf.cast(topk_mask, metrics.dtype)
            topk_mask = tf.reshape(topk_mask, [-1, 1])
            topk_mask = tf.broadcast_to(topk_mask, [b * max_det, self.topk])

        row_indices = tf.tile(tf.range(b * max_det, dtype=topk_indices.dtype)[..., None], [1, self.topk])
        scatter_indices = tf.stack([row_indices, topk_indices], -1)

        # default 0
        counts_flat = tf.scatter_nd(
            indices=scatter_indices,
            updates=topk_mask,
            shape=[b * max_det, anc_n]
        )

        counts = tf.reshape(counts_flat, [b, max_det, anc_n])
        
        return counts
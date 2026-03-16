import tensorflow as tf
from src.utils.sampler import Sampler
from src.utils.util import make_anchors, xywh2xyxy, xyxy2xywh, dist2bbox, bbox2dist
from src.utils.metric import bbox_iou

class DFLDetectionLoss():
    # loss 이름 변경해야함 구조적으로 cls, bbox, dfl 3개 적용됨
    def __init__(self, model, hyp):
        self.stride = model.modules[-1].stride
        self.nc = model.modules[-1].nc
        self.no = model.modules[-1].no
        self.reg_max = model.modules[-1].reg_max
        self.project = model.modules[-1].dfl.project

        self.sampler = Sampler(self.nc)

        self.bce = BCELoss(x_logit=True, y_logit=False)
        self.box_loss = BboxLoss()
        self.dfl_loss = DFLoss(self.reg_max)

        self.hyp = hyp

    def __call__(self, preds, gts):   
        anchors, strides = make_anchors(preds, self.stride)
        pred_shape = tf.shape(preds[0])
        img_size = tf.cast(pred_shape, tf.float32)[1:3] * self.stride[0]
        b, c = pred_shape[0], pred_shape[3]
        
        pred = tf.concat([tf.reshape(pred, [b, -1, c]) for pred in preds], 1)
        pred_dist, pred_scores = tf.split(pred, [self.reg_max*4, self.nc], axis=-1)
        pred_bboxes = self.decode_box(pred_dist, anchors)

        gt_labels, gt_bboxes = tf.split(gts, [1, 4], axis=-1)
        gt_bboxes = xywh2xyxy(gt_bboxes) * tf.tile(img_size, [2])
        mask_gt = tf.cast(tf.reduce_sum(gt_bboxes, axis=-1, keepdims=True) > 0.0, tf.float32)
    
        _, target_bboxes, target_scores, fg_mask, _ = self.sampler.sampling(tf.sigmoid(pred_scores), 
                                                                            pred_bboxes * strides, 
                                                                            anchors * strides, 
                                                                            gt_labels, 
                                                                            gt_bboxes, 
                                                                            mask_gt)
        
        norm = tf.maximum(tf.reduce_sum(target_scores), 1)
        weight = tf.reduce_sum(target_scores, -1)
        fg_weight = tf.boolean_mask(weight, fg_mask)[..., None]

        cls_loss = self.compute_cls_loss(pred_scores, target_scores) / norm
        box_loss = tf.cond(tf.reduce_any(fg_mask),
                           lambda: self.compute_box_loss(pred_bboxes, 
                                                         target_bboxes, 
                                                         strides,
                                                         fg_mask, 
                                                         fg_weight) / norm,
                           lambda: 0.0)
        dfl_loss = tf.cond(tf.reduce_any(fg_mask),
                           lambda: self.compute_dfl_loss(pred_dist, 
                                                         target_bboxes, 
                                                         anchors,
                                                         strides,
                                                         fg_mask, 
                                                         fg_weight) / norm, 
                           lambda: 0.0)

        cls_loss *= self.hyp.cls
        box_loss *= self.hyp.box
        dfl_loss *= self.hyp.dfl
        total_loss = tf.reduce_sum([cls_loss, box_loss, dfl_loss])

        b = tf.cast(b, tf.float32)
        loss_items = {"Total_Loss": total_loss,
                      "Cls_Loss":cls_loss, 
                      "Box_Loss":box_loss, 
                      "Dfl_Loss":dfl_loss}

        return total_loss * b, loss_items
    
    def compute_cls_loss(self, pred_scores, target_scores):
        return tf.reduce_sum(self.bce(pred_scores, target_scores))
    
    def compute_box_loss(self, pred_bboxes, target_bboxes, strides, fg_mask, fg_weight):
        fg_pred_bboxes = tf.boolean_mask(pred_bboxes, fg_mask)
        fg_target_bboxes = tf.boolean_mask(target_bboxes / strides, fg_mask)

        box_loss = tf.reduce_sum(self.box_loss(fg_pred_bboxes, fg_target_bboxes) * fg_weight)
        return box_loss

    def compute_dfl_loss(self, pred_dist, target_bboxes, anchors, strides, fg_mask, fg_weight):
        b = tf.shape(pred_dist)[0]
        fg_target_bboxes = tf.boolean_mask(target_bboxes / strides, fg_mask)

        if self.reg_max > 1:
            fg_pred_dist = tf.boolean_mask(pred_dist, fg_mask)
            fg_anchors = tf.boolean_mask(tf.tile(anchors[None], [b, 1, 1]), fg_mask)

            dfl_loss = tf.reduce_sum(self.dfl_loss(fg_pred_dist, fg_target_bboxes, fg_anchors) * fg_weight)
            return dfl_loss
        return 0.0
    
    def decode_box(self, pred_dist, anchors):
        if self.reg_max > 1:
            dist_shape = tf.shape(pred_dist)
            b, anc_n, c = dist_shape[0], dist_shape[1], dist_shape[2]
            pred_dist = tf.reshape(pred_dist, [b, anc_n, 4, c//4])
            pred_dist = tf.nn.softmax(pred_dist, -1)
            pred_dist = tf.matmul(pred_dist, self.project)
            pred_dist = tf.reshape(pred_dist, [b, anc_n, 4])
        return dist2bbox(pred_dist, anchors)

class BCELoss():
    def __init__(self, x_logit=True, y_logit=False, eps=1e-7):
        self.eps = eps
        if x_logit and y_logit:
            self.loss = self._loss_logit_logit
        elif x_logit and not y_logit:
            self.loss = self._loss_logit_prob
        elif not x_logit and y_logit:
            self.loss = self._loss_prob_logit
        else:
            self.loss = self._loss_prob_prob

    def __call__(self, x, y):
        return self.loss(x, y)
        
    def _loss_prob_prob(self, x, y):
        x = tf.clip_by_value(x, self.eps, 1.0 - self.eps)
        return -y * tf.math.log(x) - (1.0 - y) * tf.math.log(1.0 - x)

    def _loss_logit_prob(self, x, y):
        # tf function
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
        # general
        # term1 = tf.maximum(x, 0)
        # term2 = -x * y
        # term3 = tf.math.log(1.0 + tf.exp(-tf.abs(x)))
        # return term1 + term2 + term3
    
    def _loss_prob_logit(self, x, y):
        return self._loss_prob_prob(x, tf.sigmoid(y))
    
    def _loss_logit_logit(self, x, y):
        return self._loss_logit_prob(x, tf.sigmoid(y))
    
class BboxLoss():
    def __call__(self, pred_bboxes, gt_bboxes):
        iou = bbox_iou(pred_bboxes, gt_bboxes, xywh=False, CIoU=True)
        return (1 - iou)
        # iou_loss = tf.reduce_sum((1 - iou) * weight) / norm

        # return iou_loss
    

class DFLoss():
    def __init__(self, reg_max=16):
        self.reg_max = reg_max
        self.CE = tf.nn.sparse_softmax_cross_entropy_with_logits

    def __call__(self, pred_dist, gt_bboxes, anchor_points):
        gt_ltrb = bbox2dist(gt_bboxes, anchor_points, self.reg_max - 1)
        flat_gt_ltrb = tf.reshape(gt_ltrb, [-1])
        flat_pred_dist = tf.reshape(pred_dist, [-1, self.reg_max])

        flat_dfl_val = self._compute_dfl(flat_pred_dist, flat_gt_ltrb)
        dfl_loss = tf.reduce_mean(tf.reshape(flat_dfl_val, [-1, 4]), axis=-1, keepdims=True)

        return dfl_loss
        # dfl_loss = tf.reduce_sum(dfl_val * weight) / norm
        # return dfl_loss

    
    def _compute_dfl(self, pred_dist, gt):
        gt = tf.clip_by_value(gt, 0, self.reg_max - 1.01)
        gt_floor = tf.floor(gt)
        gt_ceil = gt_floor + 1
        wl = gt_ceil - gt
        wr = 1 - wl
        loss_floor = self.CE(logits=pred_dist, labels=tf.cast(gt_floor, tf.int32)) * wl
        loss_ceil = self.CE(logits=pred_dist, labels=tf.cast(gt_ceil, tf.int32)) * wr
        
        return loss_floor + loss_ceil 
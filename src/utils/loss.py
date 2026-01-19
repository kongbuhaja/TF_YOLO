import tensorflow as tf
from src.utils.sampler import Sampler
from src.utils.util import make_anchors, xywh2xyxy, xyxy2xywh, dist2bbox, bbox2dist
from src.utils.metric import bbox_iou
import pdb

class DFLDetectionloss():
    # loss 이름 변경해야함 구조적으로 cls, bbox, dfl 3개 적용됨
    def __init__(self, model):
        self.anchors = None
        self.stride = model.modules[-1].stride
        self.nc = model.modules[-1].nc
        self.no = model.modules[-1].no
        self.reg_max = model.modules[-1].reg_max
        self.project = model.modules[-1].dfl.project
    
        self.sampler = Sampler(self.nc)

        self.bce = BCELoss(x_logit=True, y_logit=False)
        self.bbox_loss = BboxLoss(self.reg_max)

    def __call__(self, preds, gts):
        cls_loss, box_loss, dfl_loss = 0, 0, 0

        if self.anchors is None:
            self.anchors, self.strides = make_anchors(preds, self.stride)
            self.img_size = tf.cast(tf.shape(preds[0]), tf.float32)[1:3] * self.stride[0]

        b, _, _, c = tf.shape(preds[0])
        pred = tf.concat([tf.reshape(pred, [b, -1, c]) for pred in preds], 1)
        pred_dist, pred_scores = tf.split(pred, [self.reg_max*4, self.nc], axis=-1)
        pred_bboxes = self.bbox_decode(pred_dist)

        gt_labels, gt_bboxes = tf.split(gts, [1, 4], axis=-1)
        gt_bboxes = xywh2xyxy(gt_bboxes) * tf.tile(self.img_size, [2])
        mask_gt = tf.cast(tf.reduce_sum(gt_bboxes, axis=-1, keepdims=True) > 0.0, tf.float32)
    
        _, target_bboxes, target_scores, fg_mask, _ = self.sampler.sampling(tf.sigmoid(pred_scores), 
                                                                            pred_bboxes * self.strides, 
                                                                            self.anchors * self.strides, 
                                                                            gt_labels, 
                                                                            gt_bboxes, 
                                                                            mask_gt)
        
        norm = tf.maximum(tf.reduce_sum(target_scores), 1)
        cls_loss = tf.reduce_sum(self.bce(pred_scores, target_scores)) / norm

        if tf.reduce_any(fg_mask):
            box_loss, dfl_loss = self.bbox_loss(pred_dist,
                                               pred_bboxes,
                                               self.anchors,
                                               target_bboxes / self.strides,
                                               target_scores,
                                               norm,
                                               fg_mask)


        return cls_loss, box_loss, dfl_loss

    def bbox_decode(self, pred_dist):
        if self.reg_max > 1:
            b, anc_n, c = tf.shape(pred_dist)
            pred_dist = tf.reshape(pred_dist, [b, anc_n, 4, c//4])
            pred_dist = tf.nn.softmax(pred_dist, -1)
            pred_dist @= self.project
            pred_dist = tf.reshape(pred_dist, [b, anc_n, 4])
        return dist2bbox(pred_dist, self.anchors)

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
        # general
        term1 = tf.maximum(x, 0)
        term2 = -x * y
        term3 = tf.math.log(1.0 + tf.exp(-tf.abs(x)))
        return term1 + term2 + term3
    
    def _loss_prob_logit(self, x, y):
        return self._loss_prob_prob(x, tf.sigmoid(y))
    
    def _loss_logit_logit(self, x, y):
        return self._loss_logit_prob(x, tf.sigmoid(y))
    
class BboxLoss():
    def __init__(self, reg_max):
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def __call__(self, pred_dist, pred_bboxes, anchor_points, gt_bboxes, gt_scores, norm, fg_mask):
        w = tf.reduce_sum(gt_scores, -1)
        w = tf.boolean_mask(w, fg_mask)[..., None]

        fg_pred_bboxes = tf.boolean_mask(pred_bboxes, fg_mask)
        fg_gt_bboxes = tf.boolean_mask(gt_bboxes, fg_mask)

        iou = bbox_iou(fg_pred_bboxes, fg_gt_bboxes, xywh=False, CIoU=True)
        iou_loss = tf.reduce_sum((1 - iou) * w) / norm

        dfl_loss = tf.zeros([1], tf.float32)
        if self.dfl_loss:
            gt_ltrb = bbox2dist(gt_bboxes, anchor_points, self.dfl_loss.reg_max-1)
            fg_gt_ltrb = tf.reshape(tf.boolean_mask(gt_ltrb, fg_mask), [-1])
            fg_pred_dist = tf.reshape(tf.boolean_mask(pred_dist, fg_mask), [-1, self.dfl_loss.reg_max])
            dfl_loss = self.dfl_loss(fg_pred_dist, fg_gt_ltrb)
            dfl_loss = tf.reduce_sum(dfl_loss * w) / norm
        
        return iou_loss, dfl_loss

class DFLoss():
    def __init__(self, reg_max=16):
        self.reg_max = reg_max
        self.CE = tf.nn.sparse_softmax_cross_entropy_with_logits

    def __call__(self, pred_dist, gt):
        gt = tf.clip_by_value(gt, 0, self.reg_max - 1.01)
        gt_floor = tf.floor(gt)
        gt_ceil = gt_floor + 1
        wl = gt_ceil - gt
        wr = 1 - wl
        a = self.CE(logits=pred_dist, labels=tf.cast(gt_floor, tf.int32))
        loss_floor = self.CE(logits=pred_dist, labels=tf.cast(gt_floor, tf.int32)) * wl
        loss_ceil = self.CE(logits=pred_dist, labels=tf.cast(gt_ceil, tf.int32)) * wr
        
        return tf.reduce_mean(tf.reshape(loss_floor + loss_ceil, [-1, 4]), axis=-1, keepdims=True)
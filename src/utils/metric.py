import numpy as np
import tensorflow as tf
import math

class Metrics():
    def __init__(self, nc):
        self.p, self.r, self.f1 = None, None, None
        self.aps, self.cls = None, None
        self.p_curve, self.r_curve, self.f1_curve = None, None, None
        self.x, self.prec_values = None, None
        self.nc = nc

    @property
    def ap50(self):
        return self.aps[:, 0] if self.aps is not None and len(self.aps) > 0 else 0.0
    
    @property
    def ap(self):
        return self.aps.mean(1) if self.aps is not None and len(self.aps) > 0 else 0.0
    
    @property
    def map50(self):
        return self.aps[:, 0].mean() if self.aps is not None and len(self.aps) > 0 else 0.0
    
    @property
    def map75(self):
        return self.aps[:, 5].mean() if self.aps is not None and len(self.aps) > 0 else 0.0
    
    @property
    def map(self):
        return self.aps.mean() if self.aps is not None and len(self.aps) > 0 else 0.0
    
    @property
    def maps(self):
        maps = np.zeros(self.nc) + self.map # to match mean of maps and self.map

        if self.aps is not None and len(self.aps) > 0:
            for i, c in self.cls:
                maps[c] = self.aps[i].mean()
            return maps
    
    def update(self, result):
        (self.p, self.r, self.f1,
         self.aps, self.cls,
         self.p_curve, self.r_curve, self.f1_curve,
         self.x, self.prec_values) = result

def smooth(y, f=0.05):
    nf = round(len(y) * 2 * f) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")

def compute_ap(recall, precision, method="interp"):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    elif method == "continuous":
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    
    return ap, mpre, mrec

def ap_per_class(samples=1000, eps=1e-7, **stat):
    tp = np.array(stat["tp"])
    conf = np.array(stat["conf"])
    p_cls = np.array(stat["p_cls"])
    t_cls = np.array(stat["t_cls"])
    
    i = np.argsort(-conf)
    tp, conf, p_cls = tp[i], conf[i], p_cls[i]

    unique_cls, nt = np.unique(t_cls, return_counts=True)
    nc = unique_cls.shape[0]

    x, prec_values = np.linspace(0, 1, samples), []

    if len(tp):
        ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, samples)), np.zeros((nc, samples))
        for ci, c in enumerate(unique_cls):
            i = p_cls == c
            n_l = nt[ci]
            n_p = i.sum()
            if n_p == 0 or n_l == 0:
                continue

            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_l + eps)
            r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

            precision = tpc / (tpc + fpc)
            p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    prec_values.append(np.interp(x, mrec, mpre))
        
        prec_values = np.array(prec_values) if prec_values else np.zeros((1, 1000))

        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

        if f1_curve.shape[0] > 0:
            i = smooth(f1_curve.mean(0), 0.1).argmax()
        else:
            i = 0
        p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
        return p, r, f1, ap, unique_cls.astype(int), p_curve, r_curve, f1_curve, x, prec_values
    
    else:
        p, r, f1 = np.zeros(nc), np.zeros(nc), np.zeros(nc)
        ap = np.zeros((nc, 10))
        p_curve, r_curve, f1_curve = np.zeros((nc, samples)), np.zeros((nc, samples)), np.zeros((nc, samples))
        prec_values = np.zeros((1, samples))

    return p, r, f1, ap, unique_cls.astype(int), p_curve, r_curve, f1_curve, x, prec_values

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    for loss
    box1, box2 = (..., M, 4)
    """
    if xywh:
        x1, y1, w1, h1 = np.split(box1, 4, axis=-1)
        x2, y2, w2, h2 = np.split(box2, 4, axis=-1)
        half_w1, half_h1, half_w2, half_h2 = w1/2, h1/2, w2/2, h2/2
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - half_w1, y1 - half_h1, x1 + half_w1, y1 + half_h1
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - half_w2, y2 - half_h2, x2 + half_w2, y2 + half_h2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = np.split(box1, 4, axis=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = np.split(box2, 4, axis=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    cw = np.maximum(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0)
    ch = np.maximum(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0)
    inter = cw * ch
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            p2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2)/4
            if CIoU:
                v = (4 / np.pi**2) * (np.arctan(w2/h2) - np.arctan(w1/h1))**2
                alpha = v / (v - iou + (1 + eps))
                return iou - (p2 / c2 + v * alpha)
            return iou - p2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou

def box_iou(box1, box2, xywh=True, eps=1e-7):
    """
    for val, eval
    box1: (M, 4)
    box2: (N, 4)

    Returns:
        (M, N)
    """
    if xywh:
        b1_xy, b1_wh = np.split(box1[:, None], 2, axis=-1)
        b2_xy, b2_wh = np.split(box2[None], 2, axis=-1)
        b1_half_wh, b2_half_wh = b1_wh/2, b2_wh/2
        b1_xy1, b1_xy2 = b1_xy - b1_half_wh, b1_xy + b1_half_wh
        b2_xy1, b2_xy2 = b2_xy - b2_half_wh, b2_xy + b2_half_wh
    else:
        b1_xy1, b1_xy2 = np.split(box1, 2, axis=-1)
        b2_xy1, b2_xy2 = np.split(box2, 2, axis=-1)
        b1_wh = b1_xy2 - b1_xy1 + eps
        b2_wh = b2_xy2 - b1_xy2 + eps
    
    inter = np.maximum((np.minimum(b1_xy2, b2_xy2)) - np.maximum(b1_xy1, b2_xy1), 0).prod(-1)

    return inter / (b1_wh.prod(-1) + b2_wh.prod(-1) - inter + eps)

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = tf.split(box1, 4, -1), tf.split(box2, 4, -1)
        w1_h, h1_h, w2_h, h2_h = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        box1_x1, box1_y1, box1_x2, box1_y2 = x1 - w1_h, y1 - h1_h, x1 + w1_h, y1 + h1_h
        box2_x1, box2_y1, box2_x2, box2_y2 = x2 - w2_h, y2 - h2_h, x2 + w2_h, y2 + h2_h
    else:
        box1_x1, box1_y1, box1_x2, box1_y2 = tf.split(box1, 4, -1)
        box2_x1, box2_y1, box2_x2, box2_y2 = tf.split(box2, 4, -1)
        w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1
        w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1

    w1, h1 = tf.maximum(w1, eps), tf.maximum(h1, eps)
    w2, h2 = tf.maximum(w2, eps), tf.maximum(h2, eps)

    inter = tf.maximum(tf.minimum(box1_x2, box2_x2) - tf.maximum(box1_x1, box2_x1), 0) *\
            tf.maximum(tf.minimum(box1_y2, box2_y2) - tf.maximum(box1_y1, box2_y1), 0)
    
    union = tf.maximum(w1 * h1 + w2 * h2 - inter, eps)
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = tf.maximum(tf.maximum(box1_x2, box2_x2) - tf.minimum(box1_x1, box2_x1), eps)
        ch = tf.maximum(tf.maximum(box1_y2, box2_y2) - tf.minimum(box1_y1, box2_y1), eps)
        if CIoU or DIoU:
            c2 = tf.maximum(tf.pow(cw, 2) + tf.pow(ch, 2), eps)
            rho2 = (tf.pow(box2_x1 + box2_x2 - box1_x1 - box1_x2, 2) +
                    tf.pow(box2_y1 + box2_y2 - box1_y1 - box1_y2, 2)) / 4
            if CIoU:
                v = (4 / math.pi**2) * tf.pow(tf.math.atan(w2 / h2) - tf.math.atan(w1 / h1), 2)
                alpha = v / tf.maximum(1 - iou + v, eps)
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou
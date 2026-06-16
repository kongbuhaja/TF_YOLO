import tensorflow as tf
import time

def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for feat, stride in zip(feats, strides):
        shape = tf.shape(feat)
        h, w = shape[1], shape[2]
        yv, xv = tf.meshgrid(tf.range(h, dtype=dtype), tf.range(w, dtype=dtype), indexing='ij')
        grid = tf.reshape(tf.stack((xv, yv), axis=-1), [-1, 2]) + grid_cell_offset
        anchor_points.append(grid)
        stride_tensor.append(tf.ones([h*w, 1], dtype=dtype) * stride)
    return tf.concat(anchor_points, 0), tf.concat(stride_tensor, 0)

def xywh2xyxy(xywh):
    cxy = xywh[..., :2]
    hwh = xywh[..., 2:] / 2
    return tf.concat([cxy - hwh, cxy + hwh], axis=-1)

def xyxy2xywh(xyxy):
    xy1 = xyxy[..., :2]
    xy2 = xyxy[..., 2:]
    wh = xy2 - xy1
    xy = xy1 + wh/2
    return tf.concat([xy, wh], axis=-1)
    
def dist2bbox(dist, anchor_points, xywh=True):
    lt, rb = tf.split(dist, [2, 2], axis=-1)
    xy1 = anchor_points - lt
    xy2 = anchor_points + rb
    if xywh:
        cxy = (xy1 + xy2) / 2
        wh = xy2 - xy1
        return tf.concat([cxy, wh], -1)
    return tf.concat([xy1, xy2], -1)

def bbox2dist(bbox, anchor_points, reg_max):
    xy1, xy2 = tf.split(bbox, 2, axis=-1)
    
    lt = anchor_points - xy1
    rb = xy2 - anchor_points
    
    dist = tf.concat([lt, rb], axis=-1)
    
    return tf.clip_by_value(dist, 0.0, reg_max - 0.01)

def NMS(batch_preds, 
        conf_th=0.25, 
        iou_th=0.45, 
        max_det=300, 
        nc=None, 
        max_time_img=0.05, 
        max_nms=30000, 
        max_wh=7680,
        rotated=False):
    
    pred_shape = tf.shape(batch_preds)
    b, channels = pred_shape[0], pred_shape[2]

    if rotated:
        bi = 5
        nms_func = NMS_rotated
    else:
        bi = 4
        nms_func = tf.image.non_max_suppression

    if nc == None:
        nc = channels - bi

    nm = channels - nc - bi
    mi = bi + nc

    time_limit = 2.0 + max_time_img * float(b)

    if not rotated:
        batch_boxes = xywh2xyxy(batch_preds[..., :bi])
        batch_preds = tf.concat([batch_boxes, batch_preds[..., bi:]], axis=-1)

    start_time = time.time()
    output = []

    for preds in batch_preds:
        scores = preds[:, bi:mi]

        confs = tf.reduce_max(scores, axis=-1)
        cls_ids = tf.argmax(scores, axis=-1)

        mask = confs > conf_th

        boxes = tf.boolean_mask(preds[:, :bi], mask)
        confs = tf.boolean_mask(confs, mask)[:, None]
        cls_ids = tf.cast(tf.boolean_mask(cls_ids, mask), tf.float32)[:, None]
        bgs = tf.boolean_mask(preds[:, mi:], mask)

        preds = tf.concat([boxes, confs, cls_ids, bgs], axis=-1)

        n_preds = tf.shape(preds)[0]
        if n_preds == 0:
            output.append(tf.zeros((0, 6 + nm)))
            continue

        # sorting -> tf.nms -> sort in internal code, but preds are already sorted
        sorted_indices = tf.argsort(preds[:, bi], direction="DESCENDING")
        preds = tf.gather(preds, sorted_indices[:max_nms])            

        offset = preds[:, bi+1:bi+2] * max_wh

        if rotated:
            boxes_xy = preds[:, :2] + offset
            boxes = tf.concat([boxes_xy, preds[:, 2:bi]], axis=-1)
        else:
            boxes = preds[:, :bi] + offset
        
        scores = preds[:, bi]

        nms_indices = nms_func(boxes,
                               scores,
                               max_output_size=max_det,
                               iou_threshold=iou_th)
        
        result = tf.gather(preds, nms_indices)
        output.append(result)

        if (time.time() - start_time) > time_limit:
            print(f"WARNING! NMS time limit {time_limit:.3f}s exceeded")
            break

    return output

def NMS_rotated(boxes, scores, max_output_size=300, iou_threshold=0.7):
    # boxes are already sorted
    n = tf.shape(boxes)[0]

    ious = batch_probiou(boxes, boxes)

    idx = tf.range(n)
    ri, ci = tf.meshgrid(idx, idx, indexing="ij")
    upper_mask = tf.cast(ri < ci, dtype=ious.dtype)

    ious = ious * upper_mask

    keep_mask = tf.logical_not(tf.reduce_any(ious >= iou_threshold, axis=0))

    pick = tf.squeeze(tf.where(keep_mask), axis=-1)

    return pick[:max_output_size]

def batch_probiou(obox1, obox2, eps=1e-7):
    """
    Hellinger_distance
    hd = sqrt(1 - bc) = sqrt(1 - exp(-bd))

    bc can be used instead of hd in inference for optimize, but just use hd to simplify
    """
    bd = bhattacharyya_distance(obox1, obox2)
    hd = tf.sqrt(1.0 - tf.exp(-bd) + eps)
    
    return 1.0 - hd

def bhattacharyya_distance(obox1, obox2, eps=1e-7):
    """
    Bhattacharyya distance
    BD = -ln(BC) (BC:Bhattacharyya coefficient)
    BC = integral(1/sqrt(Px, Qx)) (P, Q: probability distribution)
    PDF = exp(-(x-m)^2/(2*V)) / (sqrt(2*pi)*sd) (m: mean, V:variance, sd: standard deviation)
    Multi dimention gaussian PDF (S: covariance coefficient, D: Dimention)
    -> exp(-((x-m)^T * S^-1 * (x-m))/2) / ((2*pi)^(D/2)*sqrt(det(S)))
    -> 2D: exp(-((x-m)^T * S^-1 * (x-m))/2) / (2*pi*sqrt(det(S)))
    BC = integral(exp^(-1/4 * [(x-m1)^T * S1^-1 * (x-m1) + (x-m2)^T * S2^-1 ^ (x-m2)])) / (2*pi * sqrt(det(s1)*det(s2)))
    -> exp(-1/8(m1-m2)^T * ((S1 + S2)/2)^-1 * (m1-m2)) * sqrt(Snew)/(det(S1)*det(S2))^(-1/4) (Snew: (S1+S2)/2)
    BD = (m1 - m2)^T * (Snew/2)^-1 * (m1 - m2) + ln(det(Snew)/sqrt(det(S1) * det(S2)))/2 (S: [[a, c],[c, a]])
    
    Choose bd instead of bc for log scale
    """
    x1, y1 = tf.split(obox1[:, :2], 2, axis=-1)
    x2, y2 = tf.split(obox2[:, :2], 2, axis=-1)
    x2, y2 = tf.transpose(x2), tf.transpose(y2)

    a1, b1, c1 = get_covariance_matrix(obox1)
    a2, b2, c2 = get_covariance_matrix(obox2)
    a2, b2, c2 = tf.transpose(a2), tf.transpose(b2), tf.transpose(c2)

    a_sum = a1 + a2
    b_sum = b1 + b2
    c_sum = c1 + c2

    x_delta = x1 - x2
    y_delta = y1 - y2

    det = a_sum * b_sum - tf.square(c_sum) + eps # determinant

    t1 = (b_sum * tf.square(x_delta) + a_sum * tf.square(y_delta)) / det * 0.25
    t2 = -(c_sum * x_delta * y_delta / det) * 0.5

    det1 = tf.maximum(a1 * b1 - tf.square(c1), 0.0)
    det2 = tf.maximum(a2 * b2 - tf.square(c2), 0.0)
    det3 = tf.maximum(a_sum * b_sum - tf.square(c_sum), 0.0) / 4 + eps

    t3 = tf.math.log(det3 / (tf.sqrt(det1 * det2) + eps)) / 2

    return tf.clip_by_value(t1 + t2 + t3, eps, 100.0)

def get_covariance_matrix(obox):
    """
        Var(x) = E[x^2] - E[x]^2 = L^/12 (L=b-a, a<=x<=b)
        w2, h2 = w^2/12, h^2/12
        M = [[w2, 0], [0, h2]]: ellipse in rectangle
        RMR^T = [[a, c], [c, b]]: ellipse in rotated rectangle (R: rotate matrix)
    """
    w, h, r = tf.split(obox[:, 2:5], 3, axis=-1)

    cos = tf.cos(r)
    sin = tf.sin(r)

    w2 = tf.square(w) / 12
    h2 = tf.square(h) / 12

    a = w2 * tf.square(cos) + h2 * tf.square(sin)
    b = w2 * tf.square(sin) + h2 * tf.square(cos)
    c = (w2 - h2) * cos * sin
    return a, b, c
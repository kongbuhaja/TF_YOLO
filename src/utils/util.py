import tensorflow as tf
import math

def stop_gradient(func):
    def wrapper(*args, **kwargs):
        safe_args = tf.nest.map_structure(
            lambda x: tf.stop_gradient(x) if tf.is_tensor(x) else x, 
            args
        )
        safe_kwargs = tf.nest.map_structure(
            lambda x: tf.stop_gradient(x) if tf.is_tensor(x) else x, 
            kwargs
        )

        outputs = func(*safe_args, **safe_kwargs)
        safe_outputs = tf.nest.map_structure(
            lambda x: tf.stop_gradient(x) if tf.is_tensor(x) else x, 
            outputs
        )
        return safe_outputs
    
    return wrapper

@stop_gradient
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
    return tf.concat([cxy - hwh, cxy + hwh], -1)

def xyxy2xywh(xyxy):
    xy1 = xyxy[..., :2]
    xy2 = xyxy[..., 2:]
    wh = xy2 - xy1
    xy = xy1 + wh/2
    return tf.concat([xy, wh], -1)
    
def dist2bbox(dist, anchor_points, xywh=True):
    lt, rb = tf.split(dist, [2, 2], -1)
    xy1 = anchor_points - lt
    xy2 = anchor_points + rb
    if xywh:
        cxy = (xy1 + xy2) / 2
        wh = xy2 - xy1
        return tf.concat([cxy, wh], -1)
    return tf.concat([xy1,])

def bbox2dist(bbox, anchor_points, reg_max):
    xy1, xy2 = tf.split(bbox, 2, axis=-1)
    
    lt = anchor_points - xy1
    rb = xy2 - anchor_points
    
    dist = tf.concat([lt, rb], axis=-1)
    
    return tf.clip_by_value(dist, 0.0, reg_max - 0.01)
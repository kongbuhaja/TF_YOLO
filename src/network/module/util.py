import tensorflow as tf
import numpy as np

def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    dtype = feats[0].dtype
    for feat, stride in zip(feats, strides):
        h, w = tf.shape(feat)[1:3]
        yv, xv = tf.meshgrid(tf.range(h, dtype=dtype), tf.range(w, dtype=dtype), indexing='ij')
        grid = tf.reshape(tf.stack((xv, yv), axis=-1), [-1, 2]) + grid_cell_offset
        anchor_points.append(grid)
        stride_tensor.append(tf.ones([h*w, 1], dtype=dtype) * stride)
    return tf.concat(anchor_points), tf.concat(stride_tensor)

def dfl_decode_box(box, anchors):
    lt, rb = tf.split(box, 2, -1)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    cxcy = (x1y1 + x2y2) / 2
    wh = x2y2 - x1y1
    return tf.concat([cxcy, wh], -1)
from src.network.module.layers import *
from src.utils.util import make_anchors, dist2bbox
import tensorflow as tf
import copy

class DFL_Detect(Layer):
    def __init__(self, in_ch, out_ch, stride, light=True, e2e=False):
        super().__init__()
        self.nl = len(in_ch)
        self.nc = out_ch
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.stride = stride
        self.anchors = None
        self.e2e = e2e
        
        c2, c3 = max((16, in_ch[0] // 4, self.reg_max * 4)), max(in_ch[0], min(self.nc, 100))
        self.cv2 = ModuleList(
            Sequential(Conv(ch, c2, 3), 
                       Conv(c2, c2, 3), 
                       Conv(c2, self.reg_max * 4, 1, bn=False, act=False)) for ch in in_ch
        )
        
        if light:
            self.cv3 = ModuleList(
                Sequential(Sequential(Conv(ch, ch, 3, g=ch), Conv(ch, c3, 1)),
                           Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                           Conv(c3, self.nc, 1, bn=False, act=False)) for ch in in_ch
            )
        else:
            self.cv3 = ModuleList(
                Sequential(Conv(ch, c3, 3),
                           Conv(c3, c3, 3),
                           Conv(c3, self.nc, 1, bn=False, act=False)) for ch in in_ch
            )
            
        self.dfl = DFL(self.reg_max)
        if self.e2e:
            self.cv2e = copy.deepcopy(self.cv2)
            self.cv3e = copy.deepcopy(self.cv3)
    
    def call(self, x, training=False):
        if self.e2e:
            pass
        
        for i in range(self.nl):
            x[i] = tf.concat([self.cv2[i](x[i]), self.cv3[i](x[i])], -1)
            # Reshape 필요 b, h, w, c -> b, h*w, c 원본 살펴보자
        
        if training:
            return x
        
        return self.postprocess(x)
            
    def postprocess(self, x):
        shape = tf.shape(x[0])
        B, C = shape[0], shape[-1]
        
        if self.anchors is None:
            self.anchors, self.strides = make_anchors(x, self.stride)
        
        x = tf.concat([tf.reshape(xi, [B, -1, C]) for xi in x], 1)
        box, cls = tf.split(x, [self.reg_max * 4, self.nc], -1)
        dbox = self.decode_box(self.dfl(box), self.anchors[None]) * self.strides
        dcls = self.decode_cls(cls)
        return tf.concat([dbox, dcls], -1)
        
    def decode_box(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors)
        
    def decode_cls(self, cls):
        return tf.sigmoid(cls)
        
class Detect(Layer):
    pass
        
head_modules = {
    "detect": Detect,
    "dfl_detect": DFL_Detect,
}
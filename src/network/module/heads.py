from src.network.module.layers import *
from src.network.module.util import make_anchors, dfl_decode_box
import tensorflow as tf
import copy

class DFL_Detect(Layer):
    def __init__(self, in_ch, out_ch, strides, light=True, e2e=False):
        super().__init__()
        self.nc = out_ch
        self.nl = len(out_ch)
        self.reg_max = 16
        self.no = self.nc + self.reg_max * 4
        self.strides = strides
        self.anchors = None
        self.e2e = e2e
        
        # Why set c2, c3 using in_ch[0]?
        c2, c3 = max((16, in_ch[0] // 4, self.reg_max * 4)), max(in_ch[0], min(self.nc, 100))
        self.cv2 = ModuleList(
            Sequential(Conv(ch, c2, 3), 
                       Conv(c2, c2, 3), 
                       Conv(c2, self.reg_max * 4, 1)) for ch in in_ch
        )
        
        if light:
            self.cv3 = ModuleList(
                Sequential(Sequential(Conv(ch, ch, 3, g=ch), Conv(ch, c3, 3)),
                        Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                        Conv(c3, self.nc, 1)) for ch in in_ch
            )
        else:
            self.cv3 = ModuleList(
                Sequential(Conv(ch, c3, 3),
                           Conv(c3, c3, 3),
                           Conv(c3, self.nc, 1)) for ch in in_ch
            )
            
        self.dfl = DFL(self.reg_max)
        if self.e2e:
            self.cv2e = copy.deepcopy(self.cv2)
            self.cv3e = copy.deepcopy(self.cv3)
    
    def call(self, x, training=False):
        if self.e2e:
            pass
        
        for i in range(self.nl):
            x[i] = tf.concat([self.dfl(self.cv2[i](x[i])), self.cv3[i](x[i])], -1)
        
        if training:
            return x
        return self._inference(x)
            
    def _inference(self, x):
        B, _, _, C = tf.shape(x[0])
        if self.anchors is None:
            self.anchors, self.strides = make_anchors(x, self.strides)
        
        x = tf.concat([tf.reshape(xi, [B, -1, C]) for xi in x], 1)
        box, cls = x.split([self.reg_max * 4, self.nc], -1)
        dbox = self.decode_box(self.dfl(box), self.anchors[None]) * self.strides
        dcls = self.decode_cls(cls)
        return tf.concat([dbox, dcls], -1)
        
    def decode_box(self, bboxes, anchors):
        return dfl_decode_box(bboxes, anchors)
        
    def decode_cls(self, cls):
        return tf.sigmoid(cls)
        
        
        
        
head_modules = {
    "dfl_detect": DFL_Detect,
}
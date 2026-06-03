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
            SequentialLayer(Conv(ch, c2, 3), 
                       Conv(c2, c2, 3), 
                       Conv(c2, self.reg_max * 4, 1, bn=False, act=False)) for ch in in_ch
        )
        
        if light:
            self.cv3 = ModuleList(
                SequentialLayer(SequentialLayer(Conv(ch, ch, 3, g=ch), Conv(ch, c3, 1)),
                           SequentialLayer(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                           Conv(c3, self.nc, 1, bn=False, act=False)) for ch in in_ch
            )
        else:
            self.cv3 = ModuleList(
                SequentialLayer(Conv(ch, c3, 3),
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
        
        xi = []
        for i in range(self.nl):
            xi.append(tf.concat([self.cv2[i](x[i], training=training),
                                 self.cv3[i](x[i], training=training)], -1))
        
        if training:
            return xi
        
        return self.postprocess(xi), xi
            
    def postprocess(self, x):
        shape = tf.shape(x[0])
        B, C = shape[0], shape[-1]
        
        anchors, strides = make_anchors(x, self.stride)
        
        x = tf.concat([tf.reshape(xi, [B, -1, C]) for xi in x], 1)
        box, cls = tf.split(x, [self.reg_max * 4, self.nc], -1)
        dbox = self.decode_box(self.dfl(box), anchors[None]) * strides
        dcls = self.decode_cls(cls)
        return tf.concat([dbox, dcls], -1)
        
    def decode_box(self, bboxes, anchors):
        return dist2bbox(bboxes, anchors)
        
    def decode_cls(self, cls):
        return tf.sigmoid(cls)
    
    def initialize_bias(self):
        box_bias = 1.0
        cls_bias = -math.log((1 - 0.01) / 0.01)

        for cv2, cv3 in zip(self.cv2.layers, self.cv3.layers):
            box_conv = cv2.layers[-1].conv
            if box_conv.bias is not None:
                box_conv.bias.assign(tf.fill(box_conv.bias.shape, box_bias))

            cls_conv = cv3.layers[-1].conv
            if cls_conv.bias is not None:
                # cls_bias = (5 / self.nc / (640 / stride) ** 2) -> need input_size -> X
                cls_conv.bias.assign(tf.fill(cls_conv.bias.shape, cls_bias))
        
class Detect(Layer):
    pass
        
head_modules = {
    "detect": Detect,
    "dfl_detect": DFL_Detect,
}
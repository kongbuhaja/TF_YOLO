from src.network.module.layers import *
from src.network.module.activations import *

class C2f(Layer):
    def __init__(self, in_ch, out_ch, r=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.ch = int(out_ch * e)
        self.cv1 = Conv(in_ch, 2 * self.ch, 1, 1)
        self.cv2 = Conv((2 + r) * self.ch, out_ch, 1, 1)
        self.blocks = ModuleList(*[Bottleneck(self.ch, self.ch, shortcut, g, e=1.0) for _ in range(r)])

    def call(self, x, training=False):
        x = tf.split(self.cv1(x, training), 2, axis=-1)
        x.extend(block(x[-1], training) for block in self.blocks)
        return self.cv2(tf.concat(x, -1), training)
    
class C3(Layer):
    def __init__(self, in_ch, out_ch, r=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        ch = int(out_ch * e)
        self.cv1 = Conv(in_ch, ch, 1, 1)
        self.cv2 = Conv(in_ch, ch, 1)
        self.cv3 = Conv(2 * ch, out_ch, 1)
        self.block = Sequential(*[Bottleneck(ch, ch, shortcut, g, k=(1, 3), e=1.0) for _ in range(r)])

    def call(self, x, training=False):
        return self.cv3(tf.concat([self.block(self.cv1(x, training), training), self.cv2(x, training)], -1))

class C3k(C3):
    def __init__(self, in_ch, out_ch, r=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(in_ch, out_ch, r, shortcut, g, e)
        ch = int(out_ch * e)
        self.block = Sequential(*[Bottleneck(ch, ch, shortcut, g, k=(k, k), e=1.0) for _ in range(r)])
    
class C3k2(C2f):
    def __init__(self, in_ch, out_ch, r=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(in_ch, out_ch, r, shortcut, g, e)
        self.blocks = ModuleList(*[C3k(self.ch, self.ch, 2, shortcut, g) if c3k else Bottleneck(self.ch, self.ch, shortcut, g) for _ in range(r)])

class C2PSA(Layer):
    def __init__(self, in_ch, out_ch, r=1, dim=64, e=0.5):
        super().__init__()
        assert in_ch == out_ch
        self.ch = int(in_ch * e)
        self.cv1 = Conv(in_ch, 2 * self.ch, 1, 1)
        self.cv2 = Conv(2 * self.ch, in_ch, 1, 1)
        self.blocks = Sequential(*[PSABlock(self.ch, self.ch // dim, 0.5) for _ in range(r)])

    def call(self, x, training=False):
        a, b = tf.split(self.cv1(x, training), 2, axis=-1)
        b = self.blocks(b, training)
        return self.cv2(tf.concat([a, b], -1), training)

repeat_modules = {
    "c2f": C2f,
    "c3": C3,
    "c3k": C3k,
    "c3k2": C3k2,
    "c2psa": C2PSA,
}

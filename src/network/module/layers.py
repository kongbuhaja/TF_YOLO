import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D, Concatenate, UpSampling2D
from tensorflow.keras.initializers import HeUniform
from src.network.module.activations import *

class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        if len(layers) == 1 and not isinstance(layers[0], Layer):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def call(self, x, training=False):
        for layer in self.layers:
            x = layer(x, training=training)
        return x

class ModuleList(Layer):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = list(layers) if layers else []
        
    def __iter__(self):
        return iter(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
    
class DFL(Layer):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        # self.project = tf.constant(tf.reshape(tf.range(reg_max, dtype=tf.float32), [reg_max, 1]))
        # param 측정 때문에
        self.project = tf.Variable(tf.reshape(tf.range(reg_max, dtype=tf.float32), [reg_max, 1]), trainable=False)


    def call(self, x, training=False):
        shape = tf.shape(x)
        b, a = shape[0], shape[1]
        
        x = tf.reshape(x, [-1, 4, self.reg_max])
        x = tf.nn.softmax(x, axis=-1)
        x = tf.matmul(x, self.project)
        x = tf.reshape(x, [b, a, 4])
        return x

class Concat(Layer):
    def __init__(self, axis):
        super().__init__()
        self.concat = Concatenate(axis=axis)

    def call(self, x, training=False):
        return self.concat(x)

# class UpSample(Layer):
#     def __init__(self, size, interpolation="nearest"):
#         super().__init__()
#         self.upsample = UpSampling2D(size, interpolation=interpolation)
    
#     def call(self, x, training=False):
#         return self.upsample(x)
    
class UpSample(Layer):
    def __init__(self, size, interpolation="nearest"):
        super().__init__()
        if interpolation == "nearest":
            self.size = size
            self.upsample = self.XLAUpsample2D
        else:
            self.upsample = UpSampling2D(size, interpolation=interpolation)

    def XLAUpsample2D(self, x):
        x = tf.repeat(x, repeats=self.size, axis=1)
        x = tf.repeat(x, repeats=self.size, axis=2)
        
        return x
    
    def call(self, x, training=False):
        return self.upsample(x)

class Conv(Layer):
    default_act = activations["SiLU"]
    def __init__(self, in_ch, out_ch, k=3, s=1, p="same", g=1, d=1, bn=True, act=True):
        super().__init__()
        self.conv = Conv2D(out_ch, k, s, p,
                           groups=g,
                           dilation_rate=d,
                           use_bias=False if bn else True,
                           kernel_initializer = HeUniform())
        # regulaizer는 고민해보자
        self.bn = BatchNormalization() if bn else Identity()
        if act is True:
            self.act = self.default_act
        elif isinstance(act, str):
            self.act = activations[act]
        elif act in set(activations.values()):
            self.act = act
        else:
            self.act = Identity()

    def call(self, x, training=False):
        return self.act(self.bn(self.conv(x), training))

class Identity(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x, training=False):
        return x

class Bottleneck(Layer):
    def __init__(self, in_ch, out_ch, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        ch = int(out_ch * e)
        self.cv1 = Conv(in_ch, ch, k[0], 1)
        self.cv2 = Conv(ch, out_ch, k[1], 1, g=g)
        self.add = shortcut and in_ch == out_ch
        self.a = (ch, in_ch, out_ch)

    def call(self, x, training=False):
        if self.add:
            return x + self.cv2(self.cv1(x, training), training)
        else:
            return self.cv2(self.cv1(x, training), training)

class SPPF(Layer):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        ch = in_ch // 2
        self.cv1 = Conv(in_ch, ch, 1, 1)
        self.cv2 = Conv(in_ch * 4, out_ch, 1, 1)
        self.max_pool = MaxPool2D(pool_size=k, strides=1, padding='same')
        self.concat = Concatenate(axis=-1)

    def call(self, x, training=False):
        x = [self.cv1(x, training)]
        x.extend(self.max_pool(x[-1]) for _ in range(3))
        return self.cv2(self.concat(x), training)
    
class PSABlock(Layer):
    def __init__(self, ch, num_heads=4, attn_ratio=0.5, shortcut=True):
        super().__init__()
        self.attn = Attention(ch, num_heads, attn_ratio)
        self.ffn = Sequential([Conv(ch, ch * 2, 1), Conv(ch * 2, ch, 1, act=False)])
        self.add = shortcut

    def call(self, x, training=False):
        if self.add:
            x = x + self.attn(x, training)
            return x + self.ffn(x, training)
        else:
            return self.ffn(self.attn(x, training), training)

class Attention(Layer):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5

        qkv_dim = (self.key_dim * 2 + self.head_dim) * num_heads
        self.qkv = Conv(dim, qkv_dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
        self.proj = Conv(dim, dim, 1, act=False)

    def call(self, x, training=False):
        B, H, W, C = x.shape
        N = H * W
        
        qkv = tf.reshape(self.qkv(x, training), [B, N, self.num_heads, self.key_dim * 2 + self.head_dim])
        q, k, v = tf.split(tf.transpose(qkv, [0, 2, 1, 3]), [self.key_dim, self.key_dim, self.head_dim], axis=-1)
        
        attn = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) * self.scale, -1)
        
        x = tf.reshape(tf.transpose(tf.matmul(attn, v), perm=[0, 2, 1, 3]), [B, H, W, C])
        pe = self.pe(tf.reshape(tf.transpose(v, perm=[0, 2, 1, 3]), [B, H, W, C]), training)
        
        return self.proj(x + pe, training)

base_modules = {
    "sequential": Sequential,
    "modulelist": ModuleList,
    "conv": Conv,
    "bottleneck": Bottleneck,
    "sppf": SPPF,
    "psablock": PSABlock,
    "attention": Attention,
}

fusion_modules = {
    "concat": Concat,
}

frozen_modules = {
    "upsample": UpSample
}

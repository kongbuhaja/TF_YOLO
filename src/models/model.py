import os, math
import tensorflow as tf
from pathlib import Path
from src.network import *
import numpy as np

class Model(tf.keras.Model):
    def __init__(self, cfg):
        assert os.path.exists(cfg.file), f"Model error | unknown model {cfg.file}"
        super().__init__()
        self.model_name = cfg.name + cfg.scale
        self.weight = cfg.weight
        # self._input_shape = np.array(cfg.input_shape)
        self.modules, self.info = parse_model(cfg)
        self.build([1, *cfg.input_shape])
        self.initialize_bias()

    def call(self, x, normalize=True, training=False):
        x = self.normalize(x) if normalize else x
        y = []
        for module in self.modules:
            if module.f != -1:
                x = y[module.f] if isinstance(module.f, int) else [y[f] for f in module.f]
            x = module(x, training)
            y.append(x)
        return x
    
    def normalize(self, x):
        return tf.cast(x, tf.float32) / 255.0
    
    def initialize_bias(self):
        for module in self.modules:
            if hasattr(module, "initialize_bias"):
                module.initialize_bias()

    def extract_info(self):
        def count_params(module):
            params = 0
            for v in module.variables:
                if v.trainable:
                    params += np.size(v)
                else:
                    if "moving_mean" in v.name or "moving_variance" in v.name:
                        continue
                    params += np.size(v)
            return params
        indices = ["Idx"]
        froms = ["From"]
        repeats = ["Repeat"]
        params = ["Params"]
        modules = ["Module"]
        args = ["Args"]

        for module, arg in zip(self.modules, self.info["args"]):
            indices.append(str(module.idx))
            froms.append(str(module.f).strip("()"))
            repeats.append(str(module.r))
            params.append(str(count_params(module)))
            modules.append(str(".".join(module.__repr__().split(" ")[0].split(".")[-1:])))
            args.append(str(list(arg)))


        def get_length(target):
            # return int(np.ceil(max(len(x) for x in target)/5)*5)
            return max(len(x) for x in target)
        sheet = [indices, froms, repeats, params, modules, args]
        lengths = [get_length(indices), get_length(froms), get_length(repeats),
                   get_length(params), get_length(modules), get_length(args)]
        return sheet, lengths
            
    def summary(self, step="    "):
        sheet, lengths = self.extract_info()

        head = "📢 Model Summary"
        length = sum(lengths) + len(step)*(len(lengths)-1)
        print("="*length)
        print(" "*((length-len(head))//2) + head)
        print("="*length)
        
        for cols in zip(*sheet):
            print(step.join([f"{data:<{length}}" for data, length in zip(cols, lengths)]))

class Empty_model():
    def __call__(self, data):
        return data
    
def parse_model(cfg):
    cfg.scale  = cfg.scale if cfg.scale is not None else list(cfg.scales.keys())[0]
    depth, width, max_channel = getattr(cfg.scales, cfg.scale, "n")
    source_ch = getattr(cfg, "input_shape", [3])[-1]
    is_torch = cfg.file.parents[1].name == "torch"

    layers, channels, strides, indices, args = [], [], [], [], []
    for idx, (f, r, m, arg) in enumerate(cfg.backbone + cfg.head): # [from, repeats, module, args]
        r = max(round(r * depth), 1)
        
        module = m.lower().split(".")[-1]
        assert module in all_modules.keys(), f"{module} is wrong module"

        if isinstance(f, list):
            in_ch = tuple(channels[fn] for fn in f) if channels else tuple(ch for ch in source_ch)
            stride = tuple(strides[fn] for fn in f) if strides else tuple(1 for _ in source_ch)
            f = tuple(indices[fn] for fn in f) if indices else tuple(-1 for fn in f)
        else:
            in_ch = channels[f] if channels else source_ch
            stride = strides[f] if strides else 1
            f = indices[f] if indices else -1

        if module in (base_modules | repeat_modules).keys():
            out_ch = make_divisible(min(arg[0], max_channel) * width, 8)
            arg = [in_ch, out_ch, *arg[1:]]
            if module == "conv" and arg[3] == 2:
                stride *= 2
                
            if module in repeat_modules.keys():
                arg.insert(2, r)
                r = 1

        elif module in fusion_modules.keys():
            if module == "concat":
                if is_torch:
                    arg[0] -= 2 if arg[0] == 1 else -1

                out_ch = sum(in_ch)
                stride = stride[0]

        elif module in frozen_modules.keys():
            if module == "upsample":
                if len(arg):
                    arg.pop(0)
                stride //= 2
            out_ch = in_ch

        elif module in head_modules.keys():
            assert hasattr(cfg, "nc"), "Please set nc in your model.yaml."
            if module == "detect": # 버전별로 detect면 원래의 detect로 하고 dfl_detect면 dfl_detect로
                module = "dfl_detect"
            arg = [in_ch, getattr(cfg, "nc"), stride]
            out_ch = cfg.nc

        module = all_modules[module](*arg)
        module.idx, module.f, module.r = idx, f, r
        indices.append(idx)
        layers.append(module)
        channels.append(out_ch)
        strides.append(stride)
        args.append(arg)
        
    info = {"args": args}
        
    return layers, info

def make_divisible(x, div):
    return math.ceil(x / div) * div
    
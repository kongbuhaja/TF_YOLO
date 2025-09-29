import os, math
from pathlib import Path
from src.network import *

class Model():
    def __init__(self, file):
        name, extension = file.split(".")
        self.name = name
        if extension == "yaml":
            self.make_dir(name)

    def __call__(self, data):
        pass

    def train(self):
        self.make_dir(self.name)
        pass

    def eval(self):
        self.make_dir("eval")
        pass

    def make_dir(self, name):
        path = Path.cwd().resolve() / "results"
        os.makedirs(path, exist_ok=True)
        n = sum([d.startswith(name) for d in os.listdir(path)])
        self.path = path / f"{name}{n}"
        os.makedirs(self.path)

class Empty_model():
    def __call__(self, data):
        return data
    
def parse_model(cfg):
    name = cfg.file.name.rstrip(".yaml")
    print(name)
    scale = cfg.scale if cfg.scale is not None else list(cfg.scales.keys())[0]
    depth, width, max_channel = cfg.scales[scale]
    layers = []
    source_ch = getattr(cfg, "ch", getattr(cfg, "channel", 3))
    channels, strides = [], []

    for i, (f, r, m, args) in enumerate(cfg.backbone + cfg.head): # [from, repeats, module, args]
        r = max(round(r * depth), 1)

        module = m.lower()
        idx = module.rfind(".")
        if idx:
            package, module = module[:idx], module[idx+1:]
            if package == "nn":
                if frozen_modules[module] == frozen_modules["upsample"]:
                    args = args[1:]

        if isinstance(f, list):
            in_ch = [channels[fn] for fn in f] if channels else [ch for ch in source_ch]
            stride = [strides[fn] for fn in f] if strides else [1 for _ in source_ch]
        else:
            in_ch = channels[f] if channels else source_ch
            stride = strides[f] if strides else 1

        if module in (base_modules | repeat_modules).keys():
            out_ch = make_divisible(min(args[0], max_channel) * width, 8)
            args = [in_ch, out_ch, *args[1:]]
            if module == "conv" and args[3] == 2:
                stride *= 2
                
            if module in repeat_modules.keys():
                args.insert(2, r)
                r = 1

        elif module in fusion_modules.keys():
            if module == "concat":
                out_ch = sum(in_ch)
                stride = stride[0]

        elif module in frozen_modules.keys():
            if module == "upsample":
                stride /= 2
            out_ch = in_ch

        elif module in head_modules.keys():
            assert hasattr(cfg, "nc"), "Please set nc in your model.yaml."
            if module == "detect": # 버전별로 detect면 원래의 detect로 하고 dfl_detect면 dfl_detect로
                module = "dfl_detect"
            args = [in_ch, getattr(cfg, "nc"), stride]
            out_ch = cfg.nc

        channels.append(out_ch)
        strides.append(stride)
        layers.append([[i, module, r, args] for _ in range(r)])
        
    return layers, [source_ch] + channels, [1] + strides

def make_divisible(x, div):
    return math.ceil(x / div) * div
    
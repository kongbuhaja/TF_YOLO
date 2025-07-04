import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from yolo.data.util import coco

class Dataset():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.path = Path(self.cfg.path).resolve()

    def __call__(self, dtype):
        if not os.path.exists(self.cfg.path):
            if self.name == "coco":
                coco(self)

        return self.read_data(dtype)
    
    def read_data(self, dtype):
        data = []
        image_path = self.cfg.path / getattr(self.cfg, dtype, "")
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")
        
        disregared_count = 0
        for image_file in tqdm(os.listdir(image_path), desc=f"Reading {self.cfg.name} {dtype} data"):
            file_name, extension = image_file.split(".")
            image_file = self.cfg.path / getattr(self.cfg, dtype, "") / image_file
            label_file = self.cfg.path / getattr(self.cfg, dtype, "").replace("images", "labels") / f"{file_name}.txt"
            segments = self.read_labels(label_file)
            if segments:
                image = str(image_file)
                segments = self.split_segments(segments)
                data += [[image, *segments]]
            else:
                disregared_count += 1
        print(f"We use {len(data)} images without {disregared_count} images, which does not have labels.")
        return data

    def read_labels(self, file):
        with open(file, "r") as f:
            text = f.read()
        return text
    
    def split_segments(self, segments):
        segments = segments.split("\n")
        class_ids, coords = [], []
        for segment in segments:
            segment = segment.split(" ")
            class_ids += [segment[0]]
            coords += [np.array(segment[1:], np.float32).reshape([-1, 2])]
        class_ids = np.array(class_ids, np.float32)

        return class_ids, coords    
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from src.data.util import coco
import cv2
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class Dataset():
    def __init__(self, cfg):
        self.cfg = cfg
        self.cfg.path = Path(self.cfg.path).resolve()

    def __call__(self, dtype, cache, workers):
        if not os.path.exists(self.cfg.path):
            if self.name == "coco":
                coco(self)

        return self.read_data(dtype, cache, workers)
    
    def read_data(self, dtype, cache, workers):
        self.cache = cache
        data = []
        image_path = self.cfg.path / getattr(self.cfg, dtype, "")
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")
        self.image_path = image_path

        results = []
        image_files = os.listdir(self.image_path)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for r in tqdm(executor.map(self.read, image_files), 
                          total=len(image_files),
                          desc=f"Reading {dtype} data"):
                results.append(r)
        
        data = [r for r in results if r is not None]
        disregared_count = len(results) - len(data)
        print(f"We read {len(data)} images without {disregared_count} images, which does not have labels.")
        return data
    
    def read(self, image_file):
        file_name, extension = os.path.splitext(image_file)
        image_file = self.image_path / image_file
        label_file = self.image_path.parents[1] / "labels" / image_file.parent.name / f"{file_name}.txt"
        segments = self.read_labels(label_file)
        if segments:
            image = self.read_image(image_file) if self.cache else image_file
            segments = self.split_segments(segments)
            return [image, *segments]
        else:
            return None 

    def read_labels(self, file):
        with open(file, "r") as f:
            text = f.read()
        return text
    
    def read_image(self, file):
        return cv2.imread(file)[:, :, ::-1]
    
    def split_segments(self, segments):
        segments = segments.split("\n")
        class_ids, coords, lengths = [], [], []
        for segment in segments:
            segment = segment.split(" ")
            class_ids.append(segment[0])
            coords.append(np.array(segment[1:], np.float32).reshape([-1, 2]))
            lengths.append(len(coords[-1]))
        coords = np.concatenate(coords, axis=0)
        class_ids = np.array(class_ids, np.float32)

        return class_ids, coords, lengths
import numpy as np
import os, yaml
from pathlib import Path
from tqdm import tqdm
from src.data.util import utils
import cv2
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed

class Dataset():
    def __init__(self, cfg):
        for key, value in cfg.__dict__.items():
            setattr(self, key, value)

    def load(self, dtype, cache, workers):
        dtype = dtype.lower()
        assert dtype in ["train", "val", "eval", "test"], "only support [train, val, test]"
        
        self.util = utils[self.name]()
        if not os.path.exists(self.path):
            self.util.download(self.path, self.urls, self.dirs, workers)
        
        data = self.read_data(dtype, cache, workers)
        data["dtype"] = dtype
        return data
    
    def check_gt(self):
        if os.path.exists(self.dirs["eval"]):
            return True
        self.gt_images_json_list = []
        self.gt_annos_json_list = []
        self.gt_categories_list = [{"id": category["id"],
                                    "name": category["name"]} for category in self.category_map.values()]
        self.gt_info = {"descroption": self.name,
                        "url": self.urls,
                        "info": "automatically created gt json"}
        return False
    
    def add_gt(self, image_id, image, gts):
        image_json = {"id": image_id,
                      "file_name": f"{'0'*(12 - len(str(image_id)))}{image_id}",
                      "width": image.shape[1],
                      "height": image.shape[0]}
        self.gt_images_json_list.append(image_json)

        for class_id, box in zip(gts[:, 0], gts[:, 1:]):
            box[:2] -= box[2:]/2
            anno_json = {"id": len(self.gt_annos_json_list)+1,
                         "image_id": image_id,
                         "category_id": self.category_map[class_id]["id"],
                         "bbox": box.tolist(),
                         "area": int(np.prod(box[2:])),
                         "iscrowd": 0}
            self.gt_annos_json_list.append(anno_json)

    def save_gts(self):
        gts = {"images": self.gt_images_json_list,
               "annotations": self.gt_annos_json_list,
               "categories": self.gt_categories_list,
               "info": self.gt_info}
        self.util.save_result(self.dirs["eval"], gts)

    def eval_metric(self, pred_json_path):
        self.util.eval_metric(self.dirs["eval"], pred_json_path)

    def read_data(self, dtype, cache, workers):
        self.cache = cache
        data = []

        _dtype = "val" if dtype == "eval" else dtype
        image_dir = self.dirs[_dtype]
        if not image_dir.exists():
            raise FileNotFoundError(f"{image_dir} does not exist.")

        results = []
        image_files = os.listdir(image_dir)
        total = len(image_files)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(self.read_file, image_dir, image_file) for image_file in image_files]
                
            iterator = tqdm(as_completed(futures),
                            total=total,
                            desc=f"Reading data for {dtype}")
            for it in iterator:
                results.append(it.result())
        
        else:
            for image_file in tqdm(image_files,
                                   total=total,
                                   desc=f"Reading data for {dtype}"):
                results.append(self.read_file(image_dir, image_file))
        
        data = [r for r in results if r is not None]
        disregared_count = len(results) - len(data)
        print(f"We read {len(data)} images without {disregared_count} images, which does not have labels.")

        self.category_map = self.read_category_map()

        return {"data": data, "dtype": dtype, "category_map": self.category_map}
    
    def read_file(self, image_dir, image_file):
        image_file = image_dir / image_file
        file_name, extension = os.path.splitext(image_file.name)
        label_file = image_file.parents[2] / "labels" / image_file.parent.name / f"{file_name}.txt"
        segments = self.read_labels(label_file)

        if segments:
            image = self.read_image(image_file) if self.cache else image_file
            class_ids, coords, lengths = self.split_segments(segments)
        
            data = {"image_id": file_name,
                    "image": image,
                    "class_ids": class_ids,
                    "coords": coords,
                    "lengths": lengths}
            return data
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
        lengths = np.array(lengths, np.int32)
        
        return  class_ids, coords, lengths
    
    def read_category_map(self):
        path = self.path / "category_map.yaml"
        with open(path, encoding="utf-8") as f:
            category_map = yaml.safe_load(f.read())
        return category_map
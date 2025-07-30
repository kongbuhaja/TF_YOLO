import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import json, yaml
import numpy as np

class Data():
    def __init__(self):
        self.eval_metric = self.coco_eval

    def coco_eval(self, gt_json_path, dt_json_path, ann_type="bbox"):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        import io
        from contextlib import redirect_stdout

        summary_path = dt_json_path.parent / "evaluation.log"
        coco_gt = COCO(gt_json_path)
        coco_dt = coco_gt.loadRes(str(dt_json_path))

        coco_eval = COCOeval(coco_gt, coco_dt, ann_type)
        img_ids = sorted(coco_gt.getImgIds())
        coco_eval.params.imgIds = img_ids

        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        
        result = output_buffer.getvalue()

        with open(summary_path, "w") as f:
            f.write(result)
        print(result)

    def save_result(self, path, result):
        with open(path, "w") as f:
            json.dump(result, f)        

class COCO(Data): 
    def download(self, path, urls, dirs):
        # self._load(path, urls)
        # self._extract(path)
        # self._make_structure(path, dirs)
        data, category_map = self._parse(path)
        self._save_labels(data, dirs)
        self._save_category_map(path, category_map)

    def _load(self, path, urls):
        for dtype, url in urls.items():
            if url:
                file = path / f"{dtype}.zip"
                print(f"Downloading file for {dtype}.")

                response = requests.get(url, stream=True)
                total = int(response.headers.get("content-length", 0))
                with open(file, "wb") as file, tqdm(desc=f"{dtype}.zip",
                                                    total = total,
                                                    unit = "B",
                                                    unit_scale=True,
                                                    unit_divisor=1024) as bar:
                    for data in response.iter_content(chunk_size=16*1024):
                        size = file.write(data)
                        bar.update(size)
                print("Done.")                
            else:
                print(f"Skip {dtype} downloading.")

    def _extract(self, path):
        for file in [path for path in path.iterdir() if path.is_file()]:
            print(f"Extracting {file} to {path}.")
            with zipfile.ZipFile(file, "r") as zip:
                zip.extractall(path)
            print('Done.')
    
    def _make_structure(self, path, dirs):
        for old_dir, new_path in [("train2017", dirs["train"]),
                                 ("val2017", dirs["val"]),
                                 ("test2017", dirs["test"])]:
            old_path = path / old_dir
            label_path = new_path.replace('images', 'labels')
            os.makedirs(label_path)
            shutil.move(old_path, new_path)

    def _parse(self, path):
        data = {}
        for dtype, file in [("train", "instances_train2017.json"),
                            ("val", "instances_val2017.json")]:
            data[dtype] = {}
            path = path / "annotations" / file
            
            print(f"Realding {file}", end="")
            with open(path) as f:
                json_data = json.load(f)

            categories = {int(category["id"]): i for i, category in enumerate(json_data["categories"])}
            category_map = {i:{"id": category["id"],
                               "name": category["name"]} for i, category in enumerate(json_data["categories"])}

            for image_data in json_data["images"]:
                id = image_data["id"]
                file = image_data["file_name"]
                height, width = image_data['height'], image_data['width']
                data[dtype][id] = {"file": file,
                                   'size': (float(width), float(height)),
                                   "segments": []}
                
            for anno_data in json_data["annotations"]:
                if not anno_data["iscrowd"]:
                    image_id = anno_data["image_id"]
                    category_id = anno_data["category_id"]
                    segments = anno_data["segmentation"]
                    for segment in segments:
                        data[dtype][image_id]["segments"].append([categories[category_id],
                                                                  segment])
            print("Done")
        return data, category_map

    def _save_labels(self, data, dirs):
        for dtype in ["train", "val"]:
            image_dir = dirs[dtype]
            label_dir = image_dir.parents[1] / "labels" / image_dir.name
            print(f"Making label files for {label_dir}.")
            for image_id, anno in data[dtype].items():
                extension = anno["file"].split(".")[-1]
                label_path = label_dir / anno["file"].replace(extension, "txt")
                text = ""
                for class_id, segments in anno["segments"]:
                    segments = (np.array(segments).reshape([-1, 2]) / anno["size"]).reshape([-1])
                    text += f"{class_id}"
                    text += "".join([f" {c:.6}" for c in segments]) + "\n"
                
                with open(label_path, 'w') as f:
                    f.write(text.strip("\n"))
            print("Done.")
    
    def _save_category_map(self, path, category_map):
        path = path / "category_map.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(category_map, f)

utils = {"coco": COCO,
         "coco2": COCO}
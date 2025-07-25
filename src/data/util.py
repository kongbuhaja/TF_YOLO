import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import json, yaml
import numpy as np

def coco(cfg):
    def download():
        for dtype, url in cfg.urls.items():
            if url:
                file = cfg.path / f"{dtype}.zip"
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
    def extract():
        for file in [path for path in cfg.path.iterdir() if path.is_file()]:
            print(f"Extracting {file} to {cfg.path}.")
            with zipfile.ZipFile(file, "r") as zip:
                zip.extractall(cfg.path)
            print('Done.')
    
    def make_structure():
        for old_dir, new_dir in [("train2017", cfg.train),
                                 ("val2017", cfg.val),
                                 ("test2017", cfg.test)]:
            old_path, new_path = cfg.path / old_dir, cfg.path / new_dir
            label_path = cfg.path / new_dir.replace('images', 'labels')
            os.makedirs(label_path)
            shutil.move(old_path, new_path)

    def parse():
        data = {}
        for dtype, file in [("train", "instances_train2017.json"),
                            ("val", "instances_val2017.json")]:
            data[dtype] = {}
            path = cfg.path / "annotations" / file
            
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

    def save_labels(data):
        for dtype, image_dir in [("train", cfg.train),
                                 ("val", cfg.val)]:
            label_dir = cfg.path / image_dir.replace("images", "labels")
            print(f"Making label files for {label_dir}.")
            for image_id, anno in data[dtype].items():
                extension = anno["file"].split(".")[-1]
                path = label_dir / anno["file"].replace(extension, "txt")
                text = ""
                for class_id, segments in anno["segments"]:
                    segments = (np.array(segments).reshape([-1, 2]) / anno["size"]).reshape([-1])
                    text += f"{class_id}"
                    text += "".join([f" {c:.6}" for c in segments]) + "\n"
                
                with open(path, 'w') as f:
                    f.write(text.strip("\n"))
            print("Done.")
    
    def save_category_map(category_map):
        path = cfg.path / "category_map.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(category_map, f)

    # download()
    # extract()
    # make_structure()
    data, category_map = parse()
    save_labels(data)
    save_category_map(category_map)
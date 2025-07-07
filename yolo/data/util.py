import os
import zipfile
import shutil
import requests
from tqdm import tqdm
import json
import numpy as np

def coco(dataset):
    def download():
        for dtype, url in dataset.urls.items():
            if url:
                file = dataset.path / f"{dtype}.zip"
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
        for file in [path for path in dataset.path.iterdir() if path.is_file()]:
            print(f"Extracting {file} to {dataset.path}.")
            with zipfile.ZipFile(file, "r") as zip:
                zip.extractall(dataset.path)
            print('Done.')
    
    def make_structure():
        for old_dir, new_dir in [("train2017", dataset.train),
                                 ("val2017", dataset.val),
                                 ("test2017", dataset.test)]:
            old_path, new_path = dataset.path / old_dir, dataset.path / new_dir
            label_path = dataset.path / new_dir.replace('images', 'labels')
            os.makedirs(label_path)
            shutil.move(old_path, new_path)

    def parse():
        data = {}
        for dtype, file in [("train", "instances_train2017.json"),
                            ("val", "instances_val2017.json")]:
            data[dtype] = {}
            path = dataset.path / "annotations" / file

            with open(path) as f:
                json_data = json.load(f)

            categories = {int(category["id"]): i for i, category in enumerate(json_data["categories"])}

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
        return data

    def save_labels(data):
        for dtype, image_dir in [("train", dataset.train),
                                 ("val", dataset.val)]:
            label_dir = dataset.path / image_dir.replace("images", "labels")
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
    # download()
    # extract()
    # make_structure()
    data = parse()
    save_labels(data)
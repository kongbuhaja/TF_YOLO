import numpy as np
import os
from yolo.cfg import yaml_load
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from pathlib import Path
from tqdm import tqdm
from yolo.data.util import coco
import cv2

class Dataset():
    def __init__(self, data):
        yaml = f"{data}.yaml" if Path(data).suffix not in [".yaml", "yml"] else data
        cfg = yaml_load(yaml)
        for key, value in cfg.items():
            setattr(self, key, value)
        self.path = Path(self.path).resolve()

    def __call__(self, dtype, preload=True):
        if not os.path.exists(self.path):
            if self.name == "coco":
                coco(self)

        file_path = self.path / f"{self.name}_{dtype}.tfrecord"
        if not os.path.exists(file_path):
            data = self.scratch_data(dtype)
            self._encode_tfrecord(data, file_path, preload=preload)
        dataset = self._decode_tfrecord(file_path, preload=preload)
        return dataset
    
    def scratch_data(self, dtype):
        data = []
        image_path = self.path / getattr(self, dtype, "")
        if not image_path.exists():
            raise FileNotFoundError(f"{image_path} does not exist.")
        print(f"Scratcing {dtype} data.")
        for image_file in os.listdir(image_path):
            file_name, extension = image_file.split(".")
            image_file = self.path / getattr(self, dtype, "") / image_file
            label_file = self.path / getattr(self, dtype, "").replace("images", "labels") / f"{file_name}.txt"
            labels = self.read_labels(label_file)
            if labels:
                labels = np.array([label.split(" ") for label in labels.split("\n")], np.float32).reshape([-1, 5])
                class_ids = labels[:, 0].astype(np.int32)
                boxes = labels[:, 1:]
                data += [(str(image_file), class_ids, boxes)]
        print("Done.")

        return data

    def read_labels(self, file):
        with open(file, "r") as f:
            text = f.read()
        return text
    
    def _encode_tfrecord(self, data, file_path, preload=True):
        print(f"Encoding {file_path.name}.")
        with tf.io.TFRecordWriter(str(file_path)) as f:
            for image, class_ids, boxes in tqdm(data):
                if preload:
                    image = cv2.imread(image)[:,:,::-1]
                f.write(self._encode_data_features(image, class_ids, boxes))
        print("Done.")

    def _decode_tfrecord(self, file_path, preload=True):
        dataset = tf.data.TFRecordDataset(file_path, num_parallel_reads=AUTOTUNE)
        dataset = dataset.map(lambda x: self._decode_data_features(x, preload), num_parallel_calls=AUTOTUNE)

        return dataset

    def _decode_data_features(self, example, preload=True):
        data_features = {"image": tf.io.FixedLenFeature([], tf.string),
                         "class_ids": tf.io.VarLenFeature(tf.int64),
                         "boxes": tf.io.VarLenFeature(tf.float32)}
        example = tf.io.parse_single_example(example, data_features)
        image = example["image"] = tf.io.decode_jpeg(example["image"], channels=3) if preload else example["image"]
        class_ids = tf.reshape(tf.sparse.to_dense(example["class_ids"]), [-1])
        boxes = tf.reshape(tf.sparse.to_dense(example["boxes"]), [-1, 4])

        return image, class_ids, boxes

    def _encode_data_features(self, image, class_ids, boxes):
        data_features = {"image": self._image_feature(image) if isinstance(image, np.ndarray) else self._string_feature(image),
                         "class_ids": self._array_feature(class_ids),
                         "boxes": self._array_feature(boxes)}
        example = tf.train.Example(features=tf.train.Features(feature=data_features))
        return example.SerializeToString()

    def _image_feature(self, feature):
        return self._bytes_feature(tf.io.encode_jpeg(feature).numpy())
    
    def _array_feature(self, feature):
        if "float" in feature.dtype.name:
            return self._float_feature(np.reshape(feature, [-1]))
        elif "int" in feature.dtype.name:
            return self._int64_feature(np.reshape(feature, [-1]))
        raise Exception(f"Wrong array dtype: {feature.dtype}")
    
    def _string_feature(self, feature):
        return self._bytes_feature(feature.encode("utf-8"))
    
    def _bytes_feature(self, feature):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[feature]))
    
    def _float_feature(self, feature):
        if type(feature) == float:
            feature = [feature]
        return tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    
    def _int64_feature(self, feature):
        if type(feature) == int:
            feature = [feature]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=feature))
    
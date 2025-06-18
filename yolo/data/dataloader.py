import numpy as np
from yolo.data.dataset import Dataset
from yolo.data.augmentation import Augmentation
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.utils import Sequence
import cv2

class Dataloader(Sequence):
    def __init__(self, data):
        self.dataset = Dataset(data)
        self.preprocess = Augmentation(self.dataset.preprocess)
        self.augmentation = Augmentation(self.dataset.augmentation)

    def __call__(self, dtype, preload=True):
        dataloader = self.dataset(dtype, preload=preload)
        dataloader = dataloader.map(self.read_image) if not preload else dataloader
        dataloader = dataloader.map(self.preprocess, num_parallel_calls=AUTOTUNE)
        # dataloader = dataloader.map(self.augmentation, num_parallel_calls=AUTOTUNE) if dtype == "train" else dataloader

        return dataloader
        
    @tf.function
    def read_image(self, image, class_ids, boxes):
        image = tf.image.decode_jpeg(tf.io.read_file(image))
        return image, class_ids, boxes
    
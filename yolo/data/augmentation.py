import numpy as np
import tensorflow as tf

class Augmentation():
    def __init__(self, transforms):
        def read_transform(cfg):
            return [get_transform(transform, args) for transform, args in cfg.items() if args]
        
        self.transforms = read_transform(transforms)

    def __call__(self, image, class_ids, boxes):
        for transform in self.transforms:
            image, class_ids, boxes = transform(image, class_ids, boxes)
        return image, class_ids, boxes

class Normalize():
    def __init__(self):
        pass

    def __call__(self, image, class_ids, boxes):
        return image/255, class_ids, boxes

class Resize_padding():
    def __init__(self, image_size, random, constant=0):
        self.image_size = np.array(image_size, np.float32)
        self.random = random
        self.constant = constant

    def __call__(self, image, class_ids, boxes):
        org_size = tf.cast((tf.shape(image)[:2][::-1]), np.float32)
        ratio = self.image_size / tf.reduce_max(org_size)
        resize_size = ratio * org_size
        pad_size = self.image_size - resize_size

        pad_ratio = np.random.uniform() if self.random else 0.5
        pad_LT = tf.cast(pad_size * pad_ratio, np.int32)
        pad_RB = tf.cast(pad_size, np.int32) - pad_LT
        pad_left, pad_top = tf.unstack(pad_LT)
        pad_right, pad_bottom = tf.unstack(pad_RB)
        padding = tf.reshape(tf.stack([pad_top, pad_bottom, pad_left, pad_right, 0, 0]), [3, 2])

        new_image = tf.pad(tf.image.resize(image, tf.cast(resize_size[::-1], np.int32)), padding, constant_values=self.constant)
        mult = tf.tile(org_size * ratio, [2]) 
        add = tf.cast(tf.stack([pad_left, pad_top, 0, 0]), tf.float32)
        new_boxes = (boxes * mult + add) / tf.tile(self.image_size, [2])
        
        return new_image, class_ids, new_boxes
    
def get_transform(transform, *args):
    args = args[0]
    transform = transform.lower()

    if transform == "normalize":
        return Normalize()
    elif transform == "resize_padding":
        return Resize_padding(*args)
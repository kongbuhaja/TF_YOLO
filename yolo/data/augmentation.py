import numpy as np
import cv2

class Data_process():
    def __init__(self, transforms):
        self.transforms = []
        def read_transform(cfg):
            return [get_process(transform, args) for transform, args in cfg.items() if args]
        
        self.transforms = read_transform(transforms)

    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data
    
    def read_transform(cfg):
        for transform, args in cfg:
            process = get_process(transform, args)
    
    def update(self, epoch):
        for i in range(len(self.transforms)):
            print(self.transforms)
            self.transforms[i].update(epoch)

class Transform():
    def __init__(self):
        pass
    
    def __call__(self, *data):
        return self.transform(*data)
    
    def transform(self, *data):
        return data
    
    def update(self):
        pass

# preprocessing
class Read_image(Transform):
    def transform(self, image, class_ids, boxes):
        image = cv2.imread(image)[:, :, ::-1].astype(np.float32)
        return image, class_ids, boxes

class Normalize(Transform):
    def transform(self, image, class_ids, boxes):
        return image/255, class_ids, boxes


# ratio based augmentation
class Random_transform(Transform):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

class Random_resize(Random_transform):
    def transform(self, image, class_ids, coords):
        ratio = 1. + np.random.uniform(-self.ratio, self.ratio)
        size = np.array(image.shape[:2][::-1])
        new_size = (size * ratio).astype(np.int32)
        new_image = cv2.resize(image, new_size)
        return new_image, class_ids, coords
    
class Random_rotate(Random_transform):
    def __init__(self, degree, constant=0):
        self.degree = degree
        self.constant = constant

    def transform(self, image, class_ids, coords):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        degree = np.random.uniform(-self.degree, self.degree)
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * abs_sin + w * abs_cos), int(h * abs_cos + w * abs_sin)
    
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]

        new_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=self.constant)

        for i, coord in enumerate(coords):
            coord = np.hstack([coord * (w, h), np.ones([coord.shape[0], 1])])
            new_coord = coord @ M.T / (new_w, new_h)
            coords[i] = new_coord
    
        return new_image, class_ids, coords

class Random_translate(Random_transform):
    def transform(self, image, class_ids, coords):
        size = image.shape[:2][::-1]
        ratio = np.random.uniform(-self.ratio, self.ratio, 2)
        tx, ty = (ratio * size).astype(np.int32)
        M = np.array([[1, 0, tx],
                      [0, 1, ty]], np.float32)
        new_image = cv2.warpAffine(image, M, size)
        for i in range(len(coords)):
            coords[i] = np.clip(coords[i] + ratio, 0, 1)
        return new_image, class_ids, coords

    
# probability based augmentation
class Prob_transform(Transform):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, *data):
        if np.random.uniform() < self.prob:
            return self.transform(*data)
        return data

class Flip_ud(Prob_transform):
    def transform(self, image, class_ids, coords):
        new_image = image[::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        for i in range(len(coords)):
            coords[i] = np.stack([coords[i][:, 0],
                                  1. - coords[i][:, 1] - 1/h], -1)
        return new_image, class_ids, coords
    
class Flip_lr(Prob_transform):
    def transform(self, image, class_ids, coords):
        new_image = image[:, ::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        for i in range(len(coords)):
            coords[i] = np.stack([1. - coords[i][:, 0] - 1/w,
                                  coords[i][:, 1]], -1)
        return new_image, class_ids, coords
    
# postprocessing
class Resize_padding(Transform):
    def __init__(self, image_size, random, constant=0):
        self.image_size = np.array(image_size, np.float32)
        self.random = random
        self.constant = constant

    def transform(self, image, class_ids, coords):
        org_size = np.array(image.shape[:2][::-1], np.float32)
        ratio = self.image_size / max(org_size)
        resize_size = (ratio * org_size).astype(np.int32)
        pad_size = self.image_size.astype(np.int32) - resize_size

        pad_ratio = np.random.uniform() if self.random else 0.5
        pad_LT = (pad_size * pad_ratio).astype(np.int32)
        left, top = pad_LT
        right, bottom = left + resize_size[0], top + resize_size[1]

        new_image = np.full((*self.image_size.astype(np.int32), 3), self.constant, np.float32)
        new_image[top:bottom, left:right] = cv2.resize(image, resize_size)
        mult = org_size * ratio
        add = pad_LT.astype(np.float32)
        for i in range(len(coords)):
            coords[i] = (coords[i] * mult + add) / self.image_size

        return new_image, class_ids, coords
    
class Crop(Transform):
    def __init__(self, image_size, low, high):
        self.image_size = np.array(image_size, np.int32) if image_size is not None else None
        self.low, self.high = low, high

    def transform(self, image, class_ids, coords):
        h, w = image.shape[:2]
        org = (np.random.uniform(self.low, self.high, 2) * (w, h)).astype(np.int32)
        if self.image_size is None:
            random_size = (1 - np.random.uniform(self.low, self.high, 2)) * (w, h)
            image_size = np.min([random_size, (w, h) - org]).astype(np.int32)
        else:
            image_size = self.image_size
        
        dst = org + image_size
        new_image = image[org[1]:dst[1], org[0]:dst[0]]

        for i in range(len(coords)):
            coords[i] = np.clip((coords[i] * (w, h) - org) / image_size, 0, 1)
        return new_image, class_ids, coords

class Filter(Transform):
    def __init__(self, min_wh=10):
        self.min_wh = min_wh
    
    def transform(self, image, class_ids, coords):
        size = image.shape[:2][::-1]
        new_class_ids, new_coords = [], []
        for class_id, coord in zip(class_ids, coords):
            wh = (np.max(coord, 0) - np.min(coord, 0)) * size
            if np.all(wh > self.min_wh):
                new_class_ids += [class_id]
                new_coords += [coord]
        new_class_ids = np.array(new_class_ids)
        return image, new_class_ids, new_coords

class Segment_to_task(Transform):
    def __init__(self, task):
        self.task = task.lower()
        assert self.task in ["detect", "segment"], "Only support [\"detect\", \"segment\"]"
        if self.task == "detect":
            self.transform = self.segment_to_box
        elif self.task == "segment":
            self.transform = self.segment_to_mask

    def segment_to_box(self, image, class_ids, coords):
        boxes = []
        for coord in coords:
            xy1, xy2 = np.min(coord, 0), np.max(coord, 0)
            boxes += [np.hstack([(xy1 + xy2) / 2, (xy2 - xy1)])]
        boxes = np.array(boxes, np.float32)
        return image, (class_ids, boxes)
    
    def segment_to_mask(self, image, class_ids, coords):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.int32)
        for class_id, coord in zip(class_ids, coords):
            coord = (coord * (w, h)).astype(np.int32)
            cv2.fillPoly(mask, [coord], int(class_id + 1))
        return image, mask
    
class Batch(Transform):
    def __init__(self, args):
        self.args = args
        self.resize_padding = Resize_padding(*args["resize_padding"])
        self.strategy = Mosaic(*args["mosaic"][1:3]) if args["mosaic"][0] else self.batch_resize_padding
        self.filter = Filter(args["filter"])
        self.segment_to_task = Segment_to_task(args["segment_to_task"])
        self.mosaic_epoch = args["mosaic"][3] - args["mosaic"][4]

    def transform(self, serialized_data):
        serialized_data = self.strategy(serialized_data)
        batch_image, batch_labels = [], []
        for b, data in enumerate(serialized_data):
            data = self.filter(*data)
            image, labels = self.segment_to_task(*data)
            batch_image += [image]
            batch_labels += [labels]
        batch_image, batch_labels = self.batch(batch_image, batch_labels)
        return batch_image, batch_labels
    
    def batch_resize_padding(self, serialized_data):
        for i in range(len(serialized_data)):
            serialized_data[i] = self.resize_padding(*serialized_data[i])
        return serialized_data
    
    def batch(self, batch_image, batch_labels):
        batch_image = np.stack(batch_image, 0)
        labels = []
        if self.segment_to_task.task == "detect":
            for b, (class_ids, boxes) in enumerate(batch_labels):
                labels += [np.concatenate([np.full([*class_ids.shape, 1], b, np.float32),
                                           class_ids[:, None],
                                           boxes], -1)]
            batch_labels = np.concatenate(labels, 0)
        elif self.segment_to_task == "segment":
            batch_labels = np.stack(batch_labels, 0)
        
        return batch_image, batch_labels

    def update(self, epoch):
        if epoch > self.mosaic_epoch:
            self.strategy = self.batch_resize_padding

class Mosaic(Transform):
    def __init__(self, image_size, c_ratio):
        self.image_size = np.array(image_size, np.int32)
        self.mosaic_size = (self.image_size * 2).astype(np.int32)
        self.c_range = (0.5 - c_ratio, 0.5 + c_ratio)
        self.crop = Crop(self.image_size, 0.25 - c_ratio, 0.25 + c_ratio)

    def transform(self, serialized_data):
        mosaic_data = []
        l = len(serialized_data)
        for b in range(l):
            indices = np.random.randint(0, l, 4)
            cx, cy = (np.random.uniform(*self.c_range, 2) * self.mosaic_size).astype(np.int32)
            xys = [(0, 0, cx, cy), 
                   (cx, 0, self.mosaic_size[0], cy), 
                   (0, cy, cx, self.mosaic_size[1]), 
                   (cx, cy, *(self.mosaic_size))]
            mosaic_image = np.zeros((*(self.mosaic_size), 3), np.float32)
            mosaic_class_ids = []
            mosaic_coords = []
            for i, ((x1, y1, x2, y2), idx) in enumerate(zip(xys, indices)):
                image, class_ids, coords = serialized_data[idx]
                h, w = image.shape[:2]
                ratio = min((x2 - x1)/w, (y2 - y1)/h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                x = x1 if i % 2 else x2 - new_w
                y = y1 if i // 2 else y2 - new_h 
                mosaic_image[y:y + new_h, x:x + new_w] = cv2.resize(image, (new_w, new_h))
                for class_id, coord in zip(class_ids, coords):
                    coord = (coord * (new_w, new_h) + (x, y)) / self.mosaic_size
                    mosaic_class_ids += [class_id]
                    mosaic_coords += [coord]
            mosaic_data += [self.crop(mosaic_image, 
                                      np.array(mosaic_class_ids), 
                                      mosaic_coords)]
            
        return mosaic_data


    
def get_process(transform, *args):
    args = args[0]
    transform = transform.lower()

    if transform == "augmentation":
        return Data_process(*args)


    elif transform == "read_image":
        return Read_image()
    elif transform == "normalize":
        return Normalize()
    
    elif transform == "random_resize":
        return Random_resize(args)
    elif transform == "random_rotate":
        return Random_rotate(*args)
    elif transform == "random_translate":
        return Random_translate(args)
    elif transform == "flip_ud":
        return Flip_ud(args)
    elif transform == "flip_lr":
        return Flip_lr(args)
    
    elif transform == "resize_padding":
        return Resize_padding(*args)
    elif transform == "crop":
        return Crop(*args)
    elif transform == "filter":
        return Filter(args)
    elif transform == "segment_to_task":
        return Segment_to_task(args)
    elif transform == "batch":
        return Batch(args)
    
import numpy as np
import cv2

class Process():
    def __init__(self, cfg):
        self.transforms = [get_transform(name, args) for name, args in cfg.items() if args]

    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data
    
    def __repr__(self):
        text = ""
        for transform in self.transforms:
            text += f"{transform}\n"
        return text
    
class Postprocess(Process):
    def __init__(self, cfg):
        for name, args in cfg.items():
            if args:
                transform = get_transform(name, args)
                if name not in ["augmentation", "mosaic", "batch"]:
                    transform = get_transform("Serialized_transform", transform)
                setattr(self, name, transform)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

    def close_mosaic(self):
        self.transforms = [
            get_transform("Serialized_transform", self.augmentation),
            self.resize_padding,
            self.filter,
            self.segment_to_task,
            self.batch
        ]

    def set(self, dtype):
        if dtype == 'train':
            self.transforms = [
                self.mosaic,
                self.filter,
                self.segment_to_task,
                self.batch
            ]
        else:
            self.transforms = [
                self.resize_padding,
                self.filter,
                self.segment_to_task,
                self.batch
            ]

class Transform():
    def __init__(self):
        pass
    
    def __call__(self, *data):
        return self.apply(*data)
    
    def apply(self, *data):
        return data
    
    def __repr__(self):
        return self.__class__.__name__

class Augmentation(Transform):
    def __init__(self, cfg):
        self.transforms = [get_transform(name, args) for name, args in cfg.items() if args]
        
    def apply(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data

    def __repr__(self):
        text = "Augmentation\n"
        for transform in self.transforms:
            text += f"  {transform}\n"
        return text[:-1]

# preprocessing
class Read_image(Transform):
    def apply(self, image, class_ids, boxes):
        image = cv2.imread(image)[:, :, ::-1].astype(np.float32)
        return image, class_ids, boxes

class Normalize(Transform):
    def apply(self, image, class_ids, boxes):
        return image/255, class_ids, boxes

# augmentation
# random augmentation
class Random_transform(Transform):
    def __init__(self, ratio=0.5):
        self.ratio = ratio

class Random_HSV(Random_transform):
    def __init__(self, h_ratio, s_ratio, v_ratio):
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio

    def apply(self, image, class_ids, coords):
        rh, rs, rv = np.random.uniform(-1, 1, 3) * [self.h_ratio, self.s_ratio, self.v_ratio]
        x = np.arange(0, 256, dtype=np.uint8)
        lut_h = ((x + rh * 180) % 180).astype(np.uint8)
        lut_s = np.clip(x * (rs + 1), 0, 255).astype(np.uint8)
        lut_v = np.clip(x * (rv + 1), 0, 255).astype(np.uint8)
        lut_s[0] = 0

        h, s, v = np.split(cv2.cvtColor((image * 255).astype(np.uint8)[..., ::-1], cv2.COLOR_BGR2HSV), 3, -1)
        hsv = cv2.merge((cv2.LUT(h, lut_h),
                         cv2.LUT(s, lut_s),
                         cv2.LUT(v, lut_v)))
        new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[..., ::-1] / 255
        return new_image, class_ids, coords
        

class Random_resize(Random_transform):
    def apply(self, image, class_ids, coords):
        ratio = 1. + np.random.uniform(-self.ratio, self.ratio)
        size = np.array(image.shape[:2][::-1])
        new_size = (size * ratio).astype(np.int32)
        new_image = cv2.resize(image, new_size)
        return new_image, class_ids, coords
    
class Random_rotate(Random_transform):
    def __init__(self, degree, constant=0):
        self.degree = degree
        self.constant = constant/255

    def apply(self, image, class_ids, coords):
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        degree = np.random.uniform(-self.degree, self.degree)
        M = cv2.getRotationMatrix2D(center, degree, 1.0)
        
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * abs_sin + w * abs_cos), int(h * abs_cos + w * abs_sin)
    
        M[0, 2] += new_w / 2 - center[0]
        M[1, 2] += new_h / 2 - center[1]

        new_image = cv2.warpAffine(image, M, (new_w, new_h), borderValue=self.constant)

        new_coords = []
        for coord in coords:
            homogeneous_coord = np.hstack([coord * (w, h), np.ones([coord.shape[0], 1])])
            new_coord = homogeneous_coord @ M.T / (new_w, new_h)
            new_coords.append(new_coord)
        return new_image, class_ids, new_coords

class Random_translate(Random_transform):
    def apply(self, image, class_ids, coords):
        size = image.shape[:2][::-1]
        ratio = np.random.uniform(-self.ratio, self.ratio, 2)
        tx, ty = (ratio * size).astype(np.int32)
        M = np.array([[1, 0, tx],
                      [0, 1, ty]], np.float32)
        new_image = cv2.warpAffine(image, M, size)
        new_coords = []
        for coord in coords:
            new_coords.append(np.clip(coord + ratio, 0, 1))
        return new_image, class_ids, new_coords
    
    
# probability based augmentation
class Prob_transform(Transform):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, *data):
        if np.random.uniform() < self.prob:
            return self.apply(*data)
        return data

class Flip_ud(Prob_transform):
    def apply(self, image, class_ids, coords):
        new_image = image[::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        new_coords = []
        for coord in coords:
            new_coords.append(np.stack([coord[:, 0],
                              1. - coord[:, 1] - 1/h], -1))
        return new_image, class_ids, new_coords
    
class Flip_lr(Prob_transform):
    def apply(self, image, class_ids, coords):
        new_image = image[:, ::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        new_coords = []
        for coord in coords:
            new_coords.append(np.stack([1. - coord[:, 0] - 1/w,
                                        coord[:, 1]], -1))
        return new_image, class_ids, new_coords
    
# process
class Resize_padding(Transform):
    def __init__(self, image_size, center, constant=0):
        self.image_size = np.array(image_size, np.float32)
        self.center = center
        self.constant = constant/255

    def apply(self, image, class_ids, coords):
        org_size = np.array(image.shape[:2][::-1], np.float32)
        ratio = self.image_size / max(org_size)
        resize_size = (ratio * org_size).astype(np.int32)
        pad_size = self.image_size.astype(np.int32) - resize_size

        pad_ratio = 0.5 if self.center else np.random.uniform() 
        pad_LT = (pad_size * pad_ratio).astype(np.int32)
        left, top = pad_LT
        right, bottom = left + resize_size[0], top + resize_size[1]

        new_image = np.full((*self.image_size.astype(np.int32), 3), self.constant, np.float32)
        new_image[top:bottom, left:right] = cv2.resize(image, resize_size)
        mult = org_size * ratio
        add = pad_LT.astype(np.float32)
        new_coords = []
        for coord in coords:
            new_coords.append((coord * mult + add) / self.image_size)

        return new_image, class_ids, new_coords
    
class Crop(Transform):
    def __init__(self, image_size, low, high):
        self.image_size = np.array(image_size, np.int32) if image_size is not None else None
        self.low, self.high = low, high

    def apply(self, image, class_ids, coords):
        h, w = image.shape[:2]
        org = (np.random.uniform(self.low, self.high, 2) * (w, h)).astype(np.int32)
        if self.image_size is None:
            random_size = (1 - np.random.uniform(self.low, self.high, 2)) * (w, h)
            image_size = np.min([random_size, (w, h) - org]).astype(np.int32)
        else:
            image_size = self.image_size
        
        dst = org + image_size
        new_image = image[org[1]:dst[1], org[0]:dst[0]]

        new_coords = []
        for coord in coords:
            new_coords.append(np.clip((coord * (w, h) - org) / image_size, 0, 1))
        return new_image, class_ids, new_coords

class Filter(Transform):
    def __init__(self, min_wh=10):
        self.min_wh = min_wh
    
    def apply(self, image, class_ids, coords):
        size = image.shape[:2][::-1]
        new_class_ids, new_coords = [], []
        for class_id, coord in zip(class_ids, coords):
            wh = (np.max(coord, 0) - np.min(coord, 0)) * size
            if np.all(wh > self.min_wh):
                new_class_ids.append(class_id)
                new_coords.append(coord)
        new_class_ids = np.array(new_class_ids)
        return image, new_class_ids, new_coords

class Segment_to_task(Transform):
    def __init__(self, task):
        self.task = task.lower()
        assert self.task in ["detect", "segment"], "Only support [\"detect\", \"segment\"]"
        if self.task == "detect":
            self.apply = self.segment_to_box
        elif self.task == "segment":
            self.apply = self.segment_to_mask

    def segment_to_box(self, image, class_ids, coords):
        boxes = []
        for coord in coords:
            xy1, xy2 = np.min(coord, 0), np.max(coord, 0)
            boxes.append(np.hstack([(xy1 + xy2) / 2, (xy2 - xy1)]))
        boxes = np.array(boxes, np.float32)
        return image, (class_ids, boxes)
    
    def segment_to_mask(self, image, class_ids, coords):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.int32)
        for class_id, coord in zip(class_ids, coords):
            coord = (coord * (w, h)).astype(np.int32)
            cv2.fillPoly(mask, [coord], int(class_id + 1))
        return image, mask

# serialized transform
class Serialized_transform(Transform):
    def __init__(self, transform):
        self.transform = transform

    def apply(self, serialized_data):
        for i in range(len(serialized_data)):
            serialized_data[i] = self.transform(*serialized_data[i])
        return serialized_data
    
    def __repr__(self):
        return f"Serialized_{self.transform}"

class Mosaic(Transform):
    def __init__(self, acfg, image_size):
        self.image_size = np.array(image_size, np.int32)
        self.mosaic_size = (self.image_size * 2).astype(np.int32)
        self.c_range = (0.4, 0.6)
        self.augmentation = get_transform("augmentation", acfg)
        self.crop = get_transform("crop", [self.image_size, 0.15, 0.35])

    def apply(self, serialized_data):
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
                image, class_ids, coords = self.augmentation(*serialized_data[idx])
                h, w = image.shape[:2]
                ratio = min((x2 - x1)/w, (y2 - y1)/h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                x = x1 if i % 2 else x2 - new_w
                y = y1 if i // 2 else y2 - new_h 
                mosaic_image[y:y + new_h, x:x + new_w] = cv2.resize(image, (new_w, new_h))
                for class_id, coord in zip(class_ids, coords):
                    coord = (coord * (new_w, new_h) + (x, y)) / self.mosaic_size
                    mosaic_class_ids.append(class_id)
                    mosaic_coords.append(coord)
            
            mosaic_data.append(self.crop(mosaic_image, 
                                         np.array(mosaic_class_ids), 
                                         mosaic_coords))
        return mosaic_data
    
    def __repr__(self):
        text = f"{self.__class__.__name__}\n"
        text += f"{self.augmentation}"
        return text
    
# batch transform
class Batch(Transform):
    def __init__(self, batch_size, task):
        self.batch_size = batch_size
        self.task = task

    def apply(self, serialized_data):
        batch_image, batch_labels = [], []
        for image, labels in serialized_data:
            batch_image.append(image)
            batch_labels.append(labels)
        batch_image, batch_labels = self.batch(batch_image, batch_labels)
        return batch_image, batch_labels
    
    def batch(self, batch_image, batch_labels):
        batch_image = np.stack(batch_image, 0)
        labels = []
        if self.task == "detect":
            for b, (class_ids, boxes) in enumerate(batch_labels):
                labels.append(np.concatenate([np.full([*class_ids.shape, 1], b, np.float32),
                                              class_ids[:, None],
                                              boxes], -1))
            batch_labels = np.concatenate(labels, 0)
        elif self.task == "segment":
            batch_labels = np.stack(batch_labels, 0)
        
        return batch_image, batch_labels

def get_transform(name, args):
    name = name.lower()

    if name == "augmentation":
        transform = Augmentation(args)

    elif name == "read_image":
        transform = Read_image()
    elif name == "normalize":
        transform = Normalize()
    
    elif name == "random_hsv":
        transform = Random_HSV(*args)
    elif name == "random_resize":
        transform = Random_resize(args)
    elif name == "random_rotate":
        transform = Random_rotate(*args)
    elif name == "random_translate":
        transform = Random_translate(args)
    elif name == "flip_ud":
        transform = Flip_ud(args)
    elif name == "flip_lr":
        transform = Flip_lr(args)
    
    elif name == "resize_padding":
        transform = Resize_padding(*args)
    elif name == "crop":
        transform = Crop(*args)
    elif name == "filter":
        transform = Filter(args)
    elif name == "segment_to_task":
        transform = Segment_to_task(args)

    elif name == "serialized_transform":
        transform = Serialized_transform(args)
    elif name == 'mosaic':
        transform = Mosaic(*args)
    elif name == "batch":
        transform = Batch(*args)
    else:
        transform = None
    
    return transform
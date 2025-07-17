import numpy as np
import cv2

class Process():
    def __init__(self, cfg):
        self.transforms = []
        for name, args in cfg.items():
            transform, flag = get_transform(name, args)
            if flag:
                setattr(self, name, transform)
                self.transforms.append(transform)

    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data
    
    def __add__(self, other):
        if isinstance(other, Process):
            for name, value in other.__dict__.items():
                if hasattr(self, name):
                    s_value = getattr(self, name)
                    if isinstance(s_value, list):
                        setattr(self, name, s_value + value)
                    else:
                        return NotImplemented
                else:
                    setattr(self, name, value)
            return self
        else:
            return NotImplemented
    
    def __repr__(self):
        text = ""
        for transform in self.transforms:
            text += f"{transform}\n"
        return text
    
class Augmentation(Process):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_mosaic = hasattr(self, "mosaic")

    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data

    def close_mosaic(self):
        if hasattr(self, "mosaic"):
            self.transforms.remove(self.mosaic)
            self.use_mosaic = False

class Postprocess(Process):
    def __init__(self, cfg):
        super().__init__(cfg)

class Transform():
    def __init__(self, *args):
        pass
    
    def __call__(self, *data):
        return self.apply(*data)
    
    def apply(self, *data):
        return data
    
    def __repr__(self):
        return self.__class__.__name__

# preprocessing
class Read_image(Transform):
    """
    Read image from the path if cache is False.
    """
    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3) or path
            class_ids: np.array(m,)
            coords: squeezed coords (n, 2)
            lengths: indices of coords to unsqueeze
        output:
            new_image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        new_image = cv2.imread(image)[:,:, ::-1]
        return new_image, class_ids, coords, lengths

# augmentation
class Mosaic(Transform):
    """
    Combine 4 images to 1 mosaic image in 2x2 grid.
    """
    def __init__(self, image_size):
        """
        input:
            image_size: final image size
        """
        self.image_size = np.array(image_size, np.int32)
        self.mosaic_size = (self.image_size * 2).astype(np.int32)
        self.div = self.mosaic_size.astype(np.float32)
        self.c_range = (0.4, 0.6)
        self.crop = get_transform("crop", [self.image_size, 0.15, 0.35])[0]

    def apply(self, *serialized_data):
        """
        input:
            serialized_data: [data_1, data_2, ..., data_n]
                data_n:
                    image: np.array(h, w, 3)
                    class_ids: np.array(m,)
                    coords: np.array(n, 2) squeezed coords
                    lengths: np.array(m,) indices of coords to unsqueeze
        output:
            mosaic_image: np.array(*self.image_size, 3)
            mosaic_class_ids: np.array(nm,)
            mosaic_coords: np.array(nn, 2) squeezed coords
            mosaic_lengths: np.array(nm,) indices of coords to unsqueeze
        """
        l = len(serialized_data)
        indices = np.random.randint(0, l, 4)
        cx, cy = (np.random.uniform(*self.c_range, 2) * self.mosaic_size).astype(np.int32)
        xys = [(0, 0, cx, cy), 
                (cx, 0, self.mosaic_size[0], cy), 
                (0, cy, cx, self.mosaic_size[1]), 
                (cx, cy, *(self.mosaic_size))]
        
        mosaic_image = np.zeros((*(self.mosaic_size), 3), np.uint8)
        mosaic_class_ids = []
        mosaic_coords = []
        mosaic_lengths = []

        for i, ((x1, y1, x2, y2), idx) in enumerate(zip(xys, indices)):
            image, class_ids, coords, lengths = serialized_data[idx]
            h, w = image.shape[:2]
            ratio = min((x2 - x1)/w, (y2 - y1)/h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            x = x1 if i % 2 else x2 - new_w
            y = y1 if i // 2 else y2 - new_h 

            mult = np.array([new_w, new_h], np.float32)
            add = np.array([x,y], np.float32)
            mosaic_image[y:y + new_h, x:x + new_w] = cv2.resize(image, (new_w, new_h))
            mosaic_class_ids.append(class_ids)
            # mosaic_coords += [coord * mult + add / self.div for coord in coords]
            mosaic_coords.append((coords * mult + add) / self.div)
            mosaic_lengths.append(lengths)

        mosaic_data = self.crop(mosaic_image, 
                                np.hstack(mosaic_class_ids), 
                                np.vstack(mosaic_coords),
                                np.hstack(mosaic_lengths))
                                
        return mosaic_data
    
    
class Crop(Transform):
    """
    Crop the image based on low and high.
    """
    def __init__(self, new_image_size, low, high):
        """
        input:
            new_image_size: final image_size if it is smaller than calculated size
            low, high: ratio of the [left, top] point ([right, bottom]: min(left, top + new_image_size, image_size)
        """
        self.new_image_size = np.array(new_image_size, np.int32) if new_image_size is not None else None
        self.low, self.high = low, high

    def apply(self, image, class_ids, coords, lenghts):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(nh, nw, 3)
            class_ids: np.array(m,)
            new_coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        h, w = image.shape[:2]
        org = (np.random.uniform(self.low, self.high, 2) * (w, h)).astype(np.int32)
        if self.new_image_size is None:
            random_size = (1 - np.random.uniform(self.low, self.high, 2)) * (w, h)
            image_size = np.min([random_size, (w, h) - org]).astype(np.int32)
        else:
            image_size = self.new_image_size
        
        dst = org + image_size
        new_image = image[org[1]:dst[1], org[0]:dst[0]]

        mult = np.array([w, h], np.float32)
        add = org.astype(np.float32)
        div = image_size.astype(np.float32)
        # new_coords = [np.clip((coord * mult - add) / div, 0, 1) for coord in coords]
        new_coords = np.clip((coords * mult - add) / div, 0, 1)

        return new_image, class_ids, new_coords, lenghts
    
# random augmentation
class Random_transform(Transform):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
    
class Random_perspective(Random_transform):
    """
    Transform of perspective transform.
    """
    def __init__(self, rotate=0, translate=0, scale=0, shear=0, perspective=0, constant=0):
        """
        input:
            rotate: degree of ratation transform
            translate: ratio of translate transform
            scale: ratio of scale transform
            shear: ratio of shear transform
            perspective: ratio of perspective transform
        """
        self.rotate = rotate
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.constant = constant

    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(h, w, 3)
            class_ids: np.array(m,)
            new_coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        h, w = image.shape[:2]
        C = np.eye(3, dtype=np.float32)
        C[0, 2], C[1, 2] = -w/2, -h/2

        P = np.eye(3, dtype=np.float32)
        px, py = np.random.uniform(-self.perspective, self.perspective, 2)
        P[2, 0], P[2, 1] = px, py

        R = np.eye(3, dtype=np.float32)
        a = np.random.uniform(-self.rotate, self.rotate)
        s = 1 + np.random.uniform(-self.scale, self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        abs_cos, abs_sin = abs(R[0, 0]), abs(R[0, 1])
        new_w, new_h = int(h * abs_sin + w * abs_cos), int(h * abs_cos + w * abs_sin)

        S = np.eye(3, dtype=np.float32)
        sx, sy = np.tan(np.random.uniform(-self.shear, self.shear, 2) * np.pi / 180)
        S[0, 1], S[1, 0] = sx, sy

        T = np.eye(3, dtype=np.float32)
        tx, ty = (0.5 + np.random.uniform(-self.translate, self.translate, 2)) * (new_w, new_h)
        T[0, 2], T[1, 2] = tx, ty

        M = T @ S @ R @ P @ C

        if self.perspective:
            new_image = cv2.warpPerspective(image, M, (new_w, new_h), borderValue=self.constant)
        else:
            new_image = cv2.warpAffine(image, M[:2], dsize=(new_w, new_h), borderValue=self.constant)
        
        mult = np.array([w, h], np.float32)
        div = np.array([new_w, new_h], np.float32)
        coords = np.hstack([coords * mult, np.ones([coords.shape[0], 1], np.float32)]) @ M.T
        new_coords = np.clip(coords[:, :2] / coords[:, 2:3] / div, 0, 1)
        # new_coords = []
        # for coord in coords:
        #     coord = np.hstack([coord * mult, np.ones([coord.shape[0], 1], np.float32)]) @ M.T
        #     new_coord = coord[:, :2] / coord[:, 2:3] / div
        #     new_coords.append(np.clip(new_coord, 0, 1))
        return new_image, class_ids, new_coords, lengths
    
class Random_HSV(Random_transform):
    """
    Modulate hsv of image.
    """
    def __init__(self, h_ratio, s_ratio, v_ratio):
        """
        inputs:
            h_ratio: hue ratio
            s_ratio: saturation ratio
            v_ratio: value ratio
        """
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio

    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        rh, rs, rv = np.random.uniform(-1, 1, 3) * [self.h_ratio, self.s_ratio, self.v_ratio]
        x = np.arange(0, 256, dtype=np.uint8)
        lut_h = ((x + rh * 180) % 180).astype(np.uint8)
        lut_s = np.clip(x * (rs + 1), 0, 255).astype(np.uint8)
        lut_v = np.clip(x * (rv + 1), 0, 255).astype(np.uint8)
        lut_s[0] = 0

        h, s, v = np.split(cv2.cvtColor(image[..., ::-1], cv2.COLOR_BGR2HSV), 3, -1)
        hsv = cv2.merge((cv2.LUT(h, lut_h),
                         cv2.LUT(s, lut_s),
                         cv2.LUT(v, lut_v)))
        new_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[..., ::-1]
        return new_image, class_ids, coords, lengths
    
# probability based augmentation
class Prob_transform(Transform):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, *data):
        if np.random.uniform() < self.prob:
            return self.apply(*data)
        return data

class Flip_ud(Prob_transform):
    """
    Flip image up-down.
    """
    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(h, w, 3)
            class_ids: np.array(m,)
            new_coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        new_image = image[::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        new_coords = np.stack([coords[:, 0],
                               1. - coords[:, 1] - 1/h], -1)
        return new_image, class_ids, new_coords, lengths
    
class Flip_lr(Prob_transform):
    """
    Flip image left-right
    """
    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(h, w, 3)
            class_ids: np.array(m,)
            new_coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        new_image = image[:, ::-1]
        h, w = np.array(image.shape[:2]).astype(np.float32)
        new_coords = np.stack([1. - coords[:, 0] - 1/w,
                               coords[:, 1]], -1)
        return new_image, class_ids, new_coords, lengths
    
# process
class Resize_padding(Transform):
    """
    Resize image with padding for final image size
    """
    def __init__(self, image_size, center, constant=0):
        """
        inputs:
            image_size: final image size
            center: flag of image with center position
            constant: padding value
        """
        self.image_size = np.array(image_size, np.float32)
        self.center = center
        self.constant = constant

    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            new_image: np.array(*self.image_size, 3)
            class_ids: np.array(m,)
            new_coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        """
        org_size = np.array(image.shape[:2][::-1], np.float32)
        ratio = self.image_size / max(org_size)
        resize_size = (ratio * org_size).astype(np.int32)
        pad_size = self.image_size.astype(np.int32) - resize_size

        pad_ratio = 0.5 if self.center else np.random.uniform() 
        pad_LT = (pad_size * pad_ratio).astype(np.int32)
        left, top = pad_LT
        right, bottom = left + resize_size[0], top + resize_size[1]

        new_image = np.full((*self.image_size.astype(np.int32), 3), self.constant, np.uint8)
        new_image[top:bottom, left:right] = cv2.resize(image, resize_size)
        
        mult = org_size * ratio
        add = pad_LT.astype(np.float32)
        new_coords = (coords * mult + add) / self.image_size
        return new_image, class_ids, new_coords, lengths
    
class Unsqueeze_coords(Transform):
    """
    Unsqueeze coords (vector to list: np.array to [np.array, np.array])
    """
    def apply(self, image, class_ids, coords, lengths):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: np.array(n, 2) squeezed coords
            lengths: np.array(m,) indices of coords to unsqueeze
        output:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            new_coords: [np.array(n_1, 2), np.array(n_2, 2)]
        """
        new_coords = np.split(coords, np.cumsum(lengths)[:-1])
        return image, class_ids, new_coords

class Filter(Transform):
    """
    Filter too samll object.
    """
    def __init__(self, ratio=10):
        """
        inputs:
            ratio: ratio for filater size
        """
        self.ratio = np.array(ratio)
    
    def apply(self, image, class_ids, coords):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: [np.array(n_1, 2), np.array(n_2, 2)]
        output:
            image: np.array(h, w, 3)
            new_class_ids: np.array(nm,)
            new_coords: [np.array(nn_1, 2), np.array(nn_2, 2)]
        """
        size = image.shape[:2][::-1]
        filter_size = self.ratio * size

        mins = np.array([coord.min(axis=0) for coord in coords])
        maxs = np.array([coord.max(axis=0) for coord in coords])
        wh = (maxs - mins) * size
        
        mask = (wh > filter_size).all(axis=-1)

        new_class_ids = class_ids[mask]
        new_coords = [coords[i] for i in np.flatnonzero(mask)]

        return image, new_class_ids, new_coords

class Segment_to_task(Transform):
    """
    Transform segments to others [box, mask]
    """
    def __init__(self, task):
        """
        input:
            task = str
        """
        self.task = task.lower()
        assert self.task in ["box", "segment"], "Only support [\"box\", \"segment\"]"
        if self.task == "box":
            self.apply = self.segment_to_box
        elif self.task == "segment":
            self.apply = self.segment_to_mask

    def segment_to_box(self, image, class_ids, coords):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: [np.array(n_1, 2), np.array(n_2, 2)]
        output:
            image: np.array(h, w, 3)
            boxes: np.array(m, 5) [c, x, y, w, h]
        """
        xywh = []
        for coord in coords:
            xy1, xy2 = np.min(coord, 0), np.max(coord, 0)
            xywh.append(np.hstack([(xy1 + xy2) / 2, (xy2 - xy1)]))
        boxes = np.hstack([class_ids[:, None], np.array(xywh, np.float32)]) if class_ids.shape[0] else np.zeros([0, 6])
        return image, boxes
    
    def segment_to_mask(self, image, class_ids, coords):
        """
        input:
            image: np.array(h, w, 3)
            class_ids: np.array(m,)
            coords: [np.array(n_1, 2), np.array(n_2, 2)]
        output:
            image: np.array(h, w, 3)
            mask: np.array(h, w) with class_ids
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.int32)
        mult = np.array((w,h)).astype(np.float32)
        new_coords = [(coord * mult).astype(np.int32) for coord in coords]
        for class_id, coord in zip(class_ids, new_coords):
            cv2.fillPoly(mask, [coord], int(class_id + 1))
        return image, mask
    
# batch transform
class Batch(Transform):
    """
    Batch serialized data
    """
    def __init__(self, task):
        """
        input:
            task: str
        """
        self.task = task

    def apply(self, serialized_data):
        """
        input:
            serialized_data: [data_1, data_2, ..., data_n]
                data:  
                    image: np.array(h, w, 3)
                    labels: one of [boxes, mask]
                        boxes: np.array(n, 5) [c, x, y, w, h]
                        mask: np.array(h, w)
        output:
            batch_data: [batch_image, batch_labels]
                batch_image: np.array(b, h, w, 3)
                batch_labels: one of [batch_boxes, batch_mask]
                    batch_boxes: np.array(m, 6) [b, c, x, y, w, h]
                    batch_mask: np.array(b, h, w)
        """
        serialized_image, serialized_labels = zip(*serialized_data)

        batch_image = np.stack(serialized_image, 0).astype(np.float32)
        if self.task == "box":
            batch_labels = self.batch_boxes(serialized_labels)
        elif self.task == "segment":
            batch_labels = self.batch_masks(serialized_labels)
        return batch_image, batch_labels
    
    def batch_boxes(self, serialized_boxes):
        """
        input:
            serialized_boxes: [boxes_1, boxes_2, ..., boxes_b]
                boxes: np.array(n, 5)
        output:
            batch_labels: np.array(m, 6) [b, c, x, y, w h]
        """
        result = []
        for b, boxes in enumerate(serialized_boxes):
            n = boxes.shape[0]
            if n:
                batch_idx = np.full((n, 1), b, dtype=np.float32)
                result.append(np.hstack((batch_idx, boxes)))

        if result:
            return np.vstack(result)
        return np.empty((0, 5), dtype=np.float32)
    
    def batch_masks(self, serialized_mask):
        """
        inmput:
            serialized_mask: [mask_1, mask_2, ..., mask_b]
                mask: np.array(h, w)
        output:
            batch_mask: np.array(b, h, w)
        """
        return np.stack(serialized_mask, 0)
    
class Normalize(Transform):
    """
    Normalize image (/255)
    """
    def apply(self, image, labels):
        """
        input:
            image: np.array(b, h, w, 3) batch image
            labels: one of [batch_boxes, batch_mask]
        """
        return image/255, labels

def get_transform(name, args):
    name = name.lower()

    if name == "read_image":
        transform = Read_image()
        args = not args
    elif name == 'mosaic':
        transform = Mosaic(args)
    elif name == "crop":
        transform = Crop(*args)
    elif name == "random_perspective":
        transform = Random_perspective(*args)
        args = sum(args)
    elif name == "random_hsv":
        transform = Random_HSV(*args)
        args = sum(args)
    elif name == "flip_ud":
        transform = Flip_ud(args)
    elif name == "flip_lr":
        transform = Flip_lr(args)
    
    elif name == "resize_padding":
        transform = Resize_padding(*args)
    elif name == "unsqueeze_coords":
        transform = Unsqueeze_coords()
    elif name == "filter":
        transform = Filter(args)
    elif name == "segment_to_task":
        transform = Segment_to_task(args)
    
    elif name == "batch":
        transform = Batch(args)
    elif name == "normalize":
        transform = Normalize()
    else:
        transform = None

    return transform, bool(args)
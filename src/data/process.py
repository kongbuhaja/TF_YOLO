import numpy as np
import cv2

class Process():
    def __init__(self, cfg=None):
        self.transforms = []
        if cfg is not None:
            for name, args in cfg.items():
                transform, flag = get_transform(name, args)
                if flag:
                    setattr(self, transform.name, transform)
                    self.transforms.append(transform)

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __add__(self, other):
        if isinstance(other, Process):
            self.transforms.extend(other.transforms)
            for key, value in other.__dict__.items():
                if isinstance(value, Transform):
                    setattr(self, key, value)
            return self
        else:
            return NotImplemented
    
    def __repr__(self):
        text = ""
        for transform in self.transforms:
            text += f"{transform}\n"
        return text
    
    def close_mosaic(self):
        if hasattr(self, "mosaic"):
            self.transforms.remove(self.mosaic)
            del self.mosaic

class Transform():
    def __init__(self):
        pass
    
    def __call__(self, data):
        return self.apply(data)
    
    def apply(self, data):
        return data
    
    @property
    def name(self):
        return self.__class__.__name__.lower()

    def __repr__(self):
        return self.__class__.__name__

# preprocessing    
class Read_image(Transform):
    """
    Read image from the path if cache is False.
    """
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3) or path
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: list(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: list(m,) indices of coords to unsqueeze
        """
        if isinstance(data, list):
            if isinstance(data[0]["image"], np.ndarray):
                return data
            for i in range(len(data)):
                data[i]["image"] = self.read(data[i]["image"])
        elif isinstance(data, dict):
            if isinstance(data["image"], np.ndarray):
                return data
            data["image"] = cv2.imread(data["image"])[..., ::-1]
        return data
    
    def read(self, path):
        return cv2.imread(path)[..., ::-1]

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
        self.image_size = np.array(image_size[:2], np.int32)
        self.mosaic_size = (self.image_size * 2).astype(np.int32)
        self.div = self.mosaic_size.astype(np.float32)
        self.c_range = (0.4, 0.6)
        self.crop = get_transform("crop", [self.image_size, 0.2, 0.3])[0]

    def apply(self, serialized_data):
        """
        input:
            serialized_data: [data_1, data_2, ..., data_n]
                data_n: dict()
                    image: np.array(h, w, 3)
                    class_ids: np.array(m,)
                    coords: np.array(n, 2) squeezed coords
                    lengths: np.array(m,) indices of coords to unsqueeze
        output:
            mosaic_image: np.array(*self.image_size, 3)
            mosaic_class_ids: np.array(k,)
            mosaic_coords: np.array(l, 2) squeezed coords
            mosaic_lengths: np.array(k,) indices of coords to unsqueeze
        """
        cx, cy = (np.random.uniform(*self.c_range, 2) * self.mosaic_size).astype(np.int32)
        xys = [(0, 0, cx, cy), 
                (cx, 0, self.mosaic_size[0], cy), 
                (0, cy, cx, self.mosaic_size[1]), 
                (cx, cy, *(self.mosaic_size))]
        
        mosaic_image_id = []
        mosaic_image = np.zeros((*(self.mosaic_size), 3), np.uint8)
        mosaic_class_ids = []
        mosaic_coords = []
        mosaic_lengths = []

        for i, ((x1, y1, x2, y2), data) in enumerate(zip(xys, serialized_data)):
            h, w = data["image"].shape[:2]
            ratio = min((x2 - x1)/w, (y2 - y1)/h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            x = x1 if i % 2 else x2 - new_w
            y = y1 if i // 2 else y2 - new_h 

            mult = np.array([new_w, new_h], np.float32) / self.div
            add = np.array([x,y], np.float32) / self.div
            mosaic_image_id.append(str(data['image_id']))
            mosaic_image[y:y + new_h, x:x + new_w] = cv2.resize(data["image"], (new_w, new_h))
            mosaic_class_ids.append(data["class_ids"])
            mosaic_coords.append(data["coords"] * mult + add)
            mosaic_lengths.append(data["lengths"])
        
        data = {"image_id": "_".join(mosaic_image_id),
                "image": mosaic_image,
                "class_ids": np.hstack(mosaic_class_ids),
                "coords": np.vstack(mosaic_coords),
                "lengths": np.hstack(mosaic_lengths)}
                                
        return self.crop(data)
    
    
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
        self.crop_size = np.array(new_image_size, np.int32)[:2] if new_image_size is not None else None
        self.low, self.high = low, high

    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(*self.new_image_size, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        """
        h, w = data["image"].shape[:2]
        org_size = np.array([w, h], np.float32)
        left_top = (np.random.uniform(self.low, self.high, 2) * org_size).astype(np.int32)
        if self.crop_size is None:
            max_crop_size = org_size.astype(np.int32) - left_top
            min_crop_size = ((1 - 0.25) * org_size).astype(np.int32)
            crop_size = np.minimum(min_crop_size, max_crop_size).astype(np.int32)
        else:
            max_crop_size = org_size.astype(np.int32) - left_top
            crop_size = np.minimum(self.crop_size, max_crop_size)
        
        right_bottom = left_top + crop_size
        data["image"] = data["image"][left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

        mult = org_size / crop_size.astype(np.float32)
        add = -left_top.astype(np.float32) / crop_size.astype(np.float32)
        data["coords"] = np.clip(data["coords"] * mult + add, 0, 1)

        return data
    
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

    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        """
        h, w = data["image"].shape[:2]
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
            data["image"] = cv2.warpPerspective(data["image"], M, (new_w, new_h), borderValue=self.constant)
        else:
            data["image"] = cv2.warpAffine(data["image"], M[:2], dsize=(new_w, new_h), borderValue=self.constant)
        
        # mult = np.array([w, h], np.float32)
        # div = np.array([new_w, new_h], np.float32)
        # coords = np.hstack([data["coords"] * mult, np.ones([data["coords"].shape[0], 1], np.float32)]) @ M.T
        # data["coords"] = np.clip(coords[:, :2] / coords[:, 2:3] / div, 0, 1)
        
        coords = data["coords"] * np.array([w, h], np.float32)
        N = coords.shape[0]

        coords_hom = np.empty((N, 3), dtype=np.float32)
        coords_hom[:, :2] = coords
        coords_hom[:, 2] = 1.0
        
        coords_hom @= M.T 

        data["coords"] = np.clip(coords_hom[:, :2] / coords_hom[:, 2:3] / np.array([new_w, new_h], np.float32), 0, 1)
        return data
    
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

    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
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

        h, s, v = np.split(cv2.cvtColor(data["image"], cv2.COLOR_RGB2HSV), 3, -1)
        hsv = cv2.merge((cv2.LUT(h, lut_h),
                         cv2.LUT(s, lut_s),
                         cv2.LUT(v, lut_v)))
        data["image"] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return data
    
# probability based augmentation
class Prob_transform(Transform):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, data):
        if np.random.uniform() < self.prob:
            return self.apply(data)
        return data

class Flip_ud(Prob_transform):
    """
    Flip image up-down.
    """
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        """
        data["image"] = data["image"][::-1]
        h, w = np.array(data["image"].shape[:2]).astype(np.float32)
        flip_offset = 1. - 1./h
        data["coords"][:, 1] = flip_offset - data["coords"][:, 1]

        return data
    
class Flip_lr(Prob_transform):
    """
    Flip image left-right
    """
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        """
        data["image"] = data["image"][:, ::-1]
        h, w = np.array(data["image"].shape[:2]).astype(np.float32)
        flip_offset = 1. - 1./w
        data["coords"][:, 0] = flip_offset - data["coords"][:, 0]

        return data
    
# process
class Resize_padding(Transform):
    """
    Resize image with padding for final image size
    """
    def __init__(self, image_size, constant=0, center=True):
        """
        inputs:
            image_size: final image size
            center: flag of image with center position
            constant: padding value
        """
        self.image_size = np.array(image_size[:2], np.float32)
        self.center = center
        self.constant = constant

    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(*self.image_size, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        """
        org_size = np.array(data["image"].shape[:2][::-1], np.float32)
        if (self.image_size == org_size).all():
            return data
        
        ratio = self.image_size / max(org_size)
        resize_size = (ratio * org_size).astype(np.int32)
        pad_size = self.image_size.astype(np.int32) - resize_size

        pad_ratio = 0.5 if self.center else np.random.uniform() 
        pad_LT = (pad_size * pad_ratio).astype(np.int32)
        l, t = pad_LT
        r, b = l + resize_size[0], t + resize_size[1]

        new_image = np.full((*self.image_size.astype(np.int32), 3), self.constant, np.uint8)
        new_image[t:b, l:r] = cv2.resize(data["image"], resize_size)
        data["image"] = new_image
        
        mult = org_size * ratio / self.image_size
        add = pad_LT.astype(np.float32) / self.image_size

        data["coords"] = data["coords"] * mult + add
        return data
    
class Unsqueeze_coords(Transform):
    """
    Unsqueeze coords (vector to list: np.array to [np.array, np.array])
    """
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: [np.array(n_1, 2), np.array(n_2, 2), ..., np.array(n_m, 2)]
        """
        if data["coords"].shape[0] != 0:
            data["coords"] = np.split(data["coords"], np.cumsum(data["lengths"])[:-1])
        else:
            data["coords"] = []
        del data["lengths"]
        return data

class Filter(Transform):
    """
    Filter too samll object.
    """
    def __init__(self, ratio=10):
        """
        inputs:
            ratio: ratio for filater size
        """
        self.unsqueeze = Unsqueeze_coords()
        self.ratio = np.array(ratio)
    
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: [np.array(n_1, 2), np.array(n_2, 2)]
        output:
            data: dict()
                image: np.array(h, w, 3)
                new_class_ids: np.array(nm,)
                new_coords: [np.array(nn_1, 2), np.array(nn_2, 2), ..., np.array(nn_nm, 2)]
        """
        data = self.unsqueeze(data)

        if self.ratio:
            if len(data["coords"]) == 0:
                data["coords"] = []

            size = data["image"].shape[:2][::-1]
            if (self.ratio > 1.0).any():
                size = np.array(data["image"].shape[:2][::-1], dtype=np.float32)
                threshold = self.ratio / size
            else:
                threshold = self.ratio
            
            mins = np.array([coord.min(axis=0) for coord in data["coords"]])
            maxs = np.array([coord.max(axis=0) for coord in data["coords"]])
            
            mask = ((maxs - mins) > threshold).all(axis=-1)

            data["class_ids"] = data["class_ids"][mask]
            data["coords"] = [data["coords"][i] for i in np.flatnonzero(mask)]

        return data

class Segment_to_task(Transform):
    """
    Transform segments to others [bbox, mask]
    """
    def __init__(self, task):
        """
        input:
            task = str
        """
        self.task = task.lower()
        assert self.task in ["bbox", "segment"], "Only support [\"bbox\", \"segment\"]"
        if self.task == "bbox":
            self.transform = self.segment_to_bbox
        elif self.task == "segment":
            self.transform = self.segment_to_mask

    def apply(self, data):
        data["labels"] = self.transform(data)
        del data["class_ids"], data["coords"]
        return data

    def segment_to_bbox(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: [np.array(n_1, 2), np.array(n_2, 2), ..., np.array(n_m, 2)]
        output:
            data: dict()
                image: np.array(h, w, 3)
                boxes: np.array(m, 5) [c, x, y, w, h]
        """
        xywh = []
        for coord in data["coords"]:
            xy1, xy2 = np.min(coord, 0), np.max(coord, 0)
            xywh.append(np.hstack([(xy1 + xy2) / 2, (xy2 - xy1)]))
        bboxes = np.hstack([data["class_ids"][:, None], np.array(xywh, np.float32)]) if data["class_ids"].shape[0] else np.zeros([0, 5])
        return bboxes
    
    def segment_to_mask(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: [np.array(n_1, 2), np.array(n_2, 2), ..., np.array(n_m, 2)]
        output:
            data: dict()
                image: np.array(h, w, 3)
                mask: np.array(h, w) labeled class_ids
        """
        h, w = data["image"].shape[:2]
        mask = np.zeros((h, w), np.int32)
        mult = np.array((w,h)).astype(np.float32)
        new_coords = [(coord * mult).astype(np.int32) for coord in data["coords"]]
        for class_id, coord in zip(data["class_ids"], new_coords):
            cv2.fillPoly(mask, [coord], int(class_id + 1))
        return mask
    
# batch transform
class Batch(Transform):
    """
    Batch serialized data
    """
    def __init__(self, task, max_det=None):
        """
        input:
            task: str
        """
        self.task = task
        self.max_det = max_det

    def apply(self, serialized_data):
        """
        input:
            serialized_data: [data_1, data_2, ..., data_n]
                data: dict()
                    image: np.array(h, w, 3)
                    labels: one of [bboxes, mask]
                        bboxes: np.array(n, 5) [c, x, y, w, h]
                        mask: np.array(h, w) labeled class_ids
        output:
            batch_data: [batch_image, batch_labels]
                image: np.array(b, h, w, 3)
                labels: one of [batch_bboxes, batch_mask]
                    batch_bboxes: np.array(b, max_det, 5) [c, x, y, w, h]
                    batch_mask: np.array(b, h, w)
                info: list(b, 4) [rx(f), ry(f), l(i), t(i)] if eval or test
        """
        serialized_image = [data["image"] for data in serialized_data]

        batch_image_id = [data['image_id'] for data in serialized_data]
        batch_image = np.stack(serialized_image, 0, dtype=np.uint8)
        batch_data = {"image_id": batch_image_id,
                      "image": batch_image}
        if self.task in ["bbox", "segment"]:
            serialized_labels = [data["labels"] for data in serialized_data]
            if self.task == "bbox":
                batch_labels = self.batch_bboxes(serialized_labels)
            elif self.task == "segment":
                batch_labels = self.batch_mask(serialized_labels)
            batch_data["labels"] = batch_labels

            if "info" in serialized_data[0]:
                batch_data["info"] = [data["info"] for data in serialized_data]

        return batch_data
    
    def batch_bboxes(self, serialized_bboxes):
        """
        input:
            serialized_boxes: [bboxes_1, bboxes_2, ..., bboxes_b]
                boxes: np.array(n, 5)
        output:
            batch_bboxes: np.array(b, max_det, 5) [c, x, y, w h]
        """
        B = len(serialized_bboxes)
        lengths = [bboxes.shape[0] for bboxes in serialized_bboxes]
        max_det = self.max_det if self.max_det else max(lengths) 
        result = np.zeros([B, max_det, 5], dtype=np.float32)
        result[..., 0] = -1
        
        for b, (bboxes, length) in enumerate(zip(serialized_bboxes, lengths)):
            valid_len = min(max_det, length)
            result[b][:valid_len] = bboxes[:valid_len]
        return result
    
    def batch_mask(self, serialized_mask):
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
    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(b, h, w, 3) batch image
                labels: one of [batch_bboxes, batch_mask]
        """
        data["image"] /= 255
        return data
    
# for eval or test
class Resize_padding_with_info(Transform):
    def __init__(self, image_size, constant):
        """
        inputs:
            image_size: final image size
            constant: padding value
        """
        self.image_size = np.array(image_size[:2], np.float32)
        self.constant = constant

    def apply(self, data):
        """
        input:
            data: dict()
                image: np.array(h, w, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
        output:
            data: dict()
                image: np.array(*self.image_size, 3)
                class_ids: np.array(m,)
                coords: np.array(n, 2) squeezed coords
                lengths: np.array(m,) indices of coords to unsqueeze
                info: list(4), [rx(f), ry(f), l(i), t(i)]
        """
        org_size = np.array(data["image"].shape[:2][::-1], np.float32)
        mult = org_size
        data["coords"] = data["coords"] * mult

        if (self.image_size == org_size).all():
            data["info"] = (1.0, 1.0, 0, 0)
            return data
        
        ratio = self.image_size / max(org_size)
        resize_size = (ratio * org_size).astype(np.int32)
        l, t = ((self.image_size.astype(np.int32) - resize_size) * 0.5).astype(np.int32)
        r, b = l + resize_size[0], t + resize_size[1]

        new_image = np.full((*self.image_size.astype(np.int32), 3), self.constant, np.uint8)
        new_image[t:b, l:r] = cv2.resize(data["image"], resize_size)
        data["image"] = new_image

        data["info"] = (*ratio, l, t)

        return data
    
    def revert_box(self, box, info):
        rx, ry, l, t = info
        xy = (box[:, :2] - (l, t)) / (rx, ry)
        wh = box[:, 2:4] / (rx, ry)
        return np.hstack([xy, wh])

    def revert_mask(self, mask, info):
        pass

def get_transform(name, args):
    name = name.lower()

    if name == "read_image":
        transform = Read_image()
        # args = not args
    elif name == "mosaic":
        transform = Mosaic(*args)
    elif name == "crop":
        transform = Crop(*args)
    elif name == "random_perspective":
        transform = Random_perspective(*args)
        args = sum(args)
    elif name == "random_hsv":
        transform = Random_HSV(*args)
        args = sum(args)
    elif name == "flip_ud":
        transform = Flip_ud(*args)
    elif name == "flip_lr":
        transform = Flip_lr(*args)
    
    elif name == "resize_padding":
        transform = Resize_padding(*args)
    elif name == "unsqueeze_coords":
        transform = Unsqueeze_coords()
    elif name == "filter":
        transform = Filter(*args)
    elif name == "segment_to_task":
        transform = Segment_to_task(*args)
    
    elif name == "batch":
        transform = Batch(*args)
    elif name == "normalize":
        transform = Normalize()

    elif name == "resize_padding_with_info":
        transform = Resize_padding_with_info(*args)
    else:
        transform = None
        args = False

    return transform, bool(args)
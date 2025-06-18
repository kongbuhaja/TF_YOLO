import cv2
import numpy as np
import matplotlib.pyplot as plt

class Painter():
    def __init__(self, classes):
        self.classes = classes
        self.colors = self._get_color_maps(len(self.classes))

    def draw_box(self, image, class_ids, boxes, thickness=2):
        """
            image: normalized numpy tensor (H, W, 3)
            class_ids: numpy tensor (B,)
            boxes: normalized numpy tensor xywh (B, 4)
        """
        image = (image*255).astype(np.uint8)
        for class_id, box in zip(class_ids, boxes):
            box *= np.tile(image.shape[:2][::-1], 2)
            p1 = (box[:2] - box[2:]/2).astype(np.int32)
            p2 = (box[:2] + box[2:]/2).astype(np.int32)
            cv2.rectangle(image, p1, p2, self.colors[class_id], thickness)
            cv2.putText(image, self.classes[class_id], np.maximum(p1-[0, 3], 0), 1, 1.5, self.colors[class_id], thickness)
        return image
    
    def show(self, image):
        plt.imshow(image)
        plt.show()

    def _get_color_maps(self, length):
        np.random.seed(42)
        colors = []
        for _ in range(length):
            colors += [np.random.randint(0, 256, 3).tolist()]
        return colors
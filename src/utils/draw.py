import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys

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
        size = np.tile(image.shape[:2][::-1], 2)
        for class_id, box in zip(class_ids, boxes):
            class_id = int(class_id)
            de_box = box * size
            p1 = (de_box[:2] - de_box[2:]/2).astype(np.int32)
            p2 = (de_box[:2] + de_box[2:]/2).astype(np.int32)
            cv2.rectangle(image, p1, p2, self.colors[class_id], thickness)
            cv2.putText(image, self.classes[class_id], np.maximum(p1-[0, 3], 0), 1, 1.5, self.colors[class_id], thickness)
        return image
    
    def draw_segment(self, image, class_ids, coords, thickness=2, alpha=0.3):
        """
            image: normalized numpy tensor (H, W, 3)
            class_ids: numpy tensor (B,)
            coords: normalized numpy tensor list xy [(n, 2), ...], len(coords)=B
        """
        image = (image*255).astype(np.uint8)
        overlay, thickness = (image.copy(), cv2.FILLED) if alpha else (image, thickness)
        denormalize_mult = image.shape[:2][::-1]
        
        for class_id, coord in zip(class_ids, coords):
            de_coord = [(coord * denormalize_mult).astype(np.int32)]
            cv2.drawContours(overlay, de_coord, -1, self.colors[class_id], thickness)
            if alpha:
                image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            
        return image
    
    def show(self, image):
        plt.imshow(image)
        plt.show()

    def _get_color_maps(self, length, l=10):
        colors = []
        n = length // l

        for sv in [0.5+0.5/n * i for i in range(n+1)]:
            for h in [i/l for i in range(l)]:
                colors.append((np.array(colorsys.hsv_to_rgb(h, sv, sv))*255).astype(np.int32).tolist())

        return colors
from src.engine.handler import Handler
import numpy as np
from tqdm import tqdm

class Evaluator(Handler):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset, "eval")
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)

    def __call__(self, model=None):
        model = model if self.model is None else model
        try:
            preds_json_list = []
            is_new_gt = not self.dataset.check_gt()
            for data in tqdm(self.dataloader,
                            total=len(self.dataloader),
                            desc=f"Evaluate {self.data.name} data"):
                
                for b, (image_id, image, info) in enumerate(zip(data["image_id"], data["image"], data["info"])):
                    # 나중에 model로 이식
                    # rx, ry, l, t = info
                    # h, w = image.shape[:2]
                    # size = np.array([w, h], np.float32)
                    # org_size = (size - np.array([l, t], np.float32)*2) / (rx, ry)
                    # add = np.array([0, -l, -t, 0, 0], np.float32)
                    # mult = np.array([1, w, h, w, h], np.float32)
                    # div = np.array([1, rx, ry, rx, ry], np.float32)
                    # preds = (data["labels"][data["labels"][:, 0] == b][:, 1:] * mult + add) / div
                    preds = data["labels"][data["labels"][:, 0] == b][:, 1:]
                    # gts = (data["labels"][data["labels"][:, 0] == b][:, 1:] * mult + add) / div
                    
                    # pred
                    image_id = int(image_id)
                    for class_id, box in zip(preds[:, 0], preds[:, 1:]):
                        box[:2] -= box[2:]/2
                        pred_json = {"image_id": image_id,
                                    "category_id": self.category_map[class_id]["id"],
                                    "bbox": box.tolist(),
                                    "score": 1.0}
                        preds_json_list.append(pred_json)

                    # gt
                    if is_new_gt:
                        gts = data["labels"][data["labels"][:, 0] == b][:, 1:]
                        self.dataset.add_gt(image_id, image, gts)

            if is_new_gt:
                self.dataset.save_gts()
            
            preds_json_path = model.path / "predictions.json"
            self.dataset.util.save_result(preds_json_path, preds_json_list)
            self.dataset.eval_metric(preds_json_path)
            
                    # import matplotlib.pyplot as plt
                    # import pdb
                    # import cv2
                    # image = (image*255).astype(np.uint8)
                    # image = image[int(t):int(h-t), int(l):int(w-l)]
                    # image = cv2.resize(image, org_size.astype(np.int32))
                    # for x,y,w,h in labels[:, 1:]:
                    #     x1, y1 = int(x - w/2), int(y - h/2)
                    #     x2, y2 = int(x + w/2), int(y + h/2)
                    #     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    # plt.imshow(image)
                    # plt.show()
                    # pdb.set_trace()
        finally:
            self.dataloader.on_epoch_end()
    
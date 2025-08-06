from src.engine.handler import Handler
from src.utils.metric import ap_per_class, box_iou, Metrics
import numpy as np
from tqdm import tqdm

class Validator(Handler):
    def __init__(self, cfg, dataset):
        super().__init__(cfg, dataset, "val")
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.metric = Metrics(len(self.dataset.classes))

    def __call__(self, model="test"):
        try:
            stats = {"tp": [],
                    "conf": [],
                    "p_cls": [],
                    "t_cls": []}
        
            for data in tqdm(self.dataloader, 
                            total=len(self.dataloader),
                            desc=f"Validate {self.dataset.name} data"):
                # continue
                batch_image, batch_labels = data["image"], data["labels"]
                if isinstance(model, str):
                    pass
                    preds = self.test_model(batch_image, batch_labels)
                else:
                    preds = model(batch_image)
                for b, pred in enumerate(preds):
                    labels = batch_labels[batch_labels[:, 0] == b]
                    t_cls, t_boxes = labels[:, 1], labels[:, 2:]
                    p_boxes, p_cls, conf = pred[:, :4], pred[:, 4], pred[:, 5]

                    iou = box_iou(t_boxes, p_boxes)
                    tp = self.match(iou, t_cls, p_cls)

                    stats["tp"].append(tp)
                    stats["conf"].append(conf)
                    stats["p_cls"].append(p_cls)
                    stats["t_cls"].append(t_cls)

            for key, value in stats.items():
                stats[key] = np.concatenate(value, 0)

            result = ap_per_class(stats["tp"], stats["conf"], stats["p_cls"], stats["t_cls"])
            self.metric.update(result)
        finally:
            self.dataloader.on_epoch_end()

    def test_model(self, images, labels):
        batch, classes, boxes = np.split(labels, [1,2], -1)
        new_labels = []
        for b in range(images.shape[0]):
            batch_mask = batch.squeeze() == b
            new_labels.append(np.concatenate([boxes[batch_mask] + 0.0, 
                                              classes[batch_mask],
                                              np.random.uniform(0.5, 1.0, (sum(batch_mask), 1))], -1))
        return new_labels
    
    def match(self, iou, gt_classes, p_classes):
        correct = np.zeros((p_classes.shape[0], self.iouv.shape[0])).astype(bool)
        correct_mask = p_classes == gt_classes[:, None]
        iou = iou * correct_mask
        for i, threshold in enumerate(self.iouv):
            matches = np.nonzero(iou >= threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1], i] = True
        return correct

        


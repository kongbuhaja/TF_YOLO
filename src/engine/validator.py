from src.engine.handler import Handler
from src.utils.metric import ap_per_class, box_iou, Metrics
from src.utils.util import NMS
import numpy as np
from src.utils.progress import ProgressBar
from src.utils.loss import DFLDetectionLoss

class Validator(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "val")
        self.loss = DFLDetectionLoss(model, cfg.loss)
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.metric = Metrics(len(self.dataset.classes))
# logs: 
#   train: [Epoch, GPU_Usage, Cls_Loss, Box_Loss, Dfl_Loss, Total_Loss, Lr]
#   val: [Images, Instances, Precision, Recall, mAP50, mAP50:95, Total_Loss]
#   eval: []
  
    def validate(self):
        def log_update():
            self.logger
        try:
            self.on_epoch_start()

            for data in ProgressBar(self.dataloader,
                                    task=self.task.upper(),
                                    headers=self.logger.keys):
                batch_image, batch_labels = data["image"], data["labels"]

                self.validate_step(batch_image, batch_labels)

            result = ap_per_class(**self.stats)
            self.metric.update(result)
        except Exception as e:
            print(f"Training loop iterrupted: {e}")
            raise e

    def validate_step(self, batch_image, batch_labels):
        def match(iou, t_cls, p_cls):
            correct = np.zeros((p_cls.shape[0], self.iouv.shape[0])).astype(bool)
            correct_mask = p_cls == t_cls[:, None]
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
        
        preds, raw_preds = self.model(batch_image)

        total_loss, loss_items = self.loss(raw_preds, batch_labels)

        if self.model.end2end:
            nms_preds = preds
        else:
            nms_preds = NMS(preds,
                            conf_th=0.001,
                            iou_th=self.cfg.iou_th,
                            max_det=self.cfg.max_det,
                            nc=self.model.nc)

        for b, pred in enumerate(nms_preds):
            labels, pred = batch_labels[b].numpy(), pred.numpy()
            t_cls, t_boxes = labels[:, 1], labels[:, 2:]
            p_boxes, conf, p_cls = pred[:, :4], pred[:, 4], pred[:, 5]

            iou = box_iou(t_boxes, p_boxes)
            tp = match(iou, t_cls, p_cls)

            self.stats["tp"].extend(tp)
            self.stats["conf"].extend(conf)
            self.stats["p_cls"].extend(p_cls)
            self.stats["t_cls"].extend(t_cls)
            
    def on_epoch_start(self):
        self.stats = {"tp": [],
                      "conf": [],
                      "p_cls": [],
                      "t_cls": []}
        
    def on_epoch_end(self):
        pass


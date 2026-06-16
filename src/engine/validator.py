from src.engine.handler import Handler
from src.utils.metric import ap_per_class, box_iou, Metrics
from src.utils.util import NMS, xywh2xyxy
import numpy as np
import tensorflow as tf

class Validator(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "val")
        self.loss = None
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.metric = Metrics(len(self.dataset.classes))
        self.val_loss_items = {}
        self.val_metrics = {}

    def validate(self):
        self.loss = self._build_loss()

        try:
            self._on_epoch_start()

            for data in self.pbar:
                batch_image, batch_labels = data["image"], data["labels"]
                total_loss, loss_items = self._validate_step(batch_image, batch_labels)
                self._on_iteration_end(loss_items, batch_labels)

            self._on_epoch_end()

        except Exception as e:
            print(f"Validation loop interrupted: {e}")
            raise e

        return self.val_loss_items, self.val_metrics

    def _build_loss(self):
        from src.utils.loss import DFLDetectionLoss
        return DFLDetectionLoss(self.model, self.cfg.loss)

    # @tf.function
    def _validate_step(self, batch_image, batch_labels):
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

        preds, raw_preds = self.model(batch_image, training=False)

        total_loss, loss_items = self.loss(raw_preds, batch_labels)

        if self.model.e2e:
            nms_preds = preds
        else:
            nms_preds = NMS(preds,
                            conf_th=0.001,
                            iou_th=self.cfg.iou_th,
                            max_det=self.cfg.max_det,
                            nc=self.model.nc)

        whwh = np.tile(self.cfg.input_shape[:2], 2)

        for b, pred in enumerate(nms_preds):
            labels, pred = batch_labels[b], pred.numpy()
            labels = labels[np.sum(labels[..., 1:5], -1) > 0]

            t_cls, t_boxes = labels[:, 0].astype(int), xywh2xyxy(labels[:, 1:]) * whwh
            p_boxes, conf, p_cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)
            iou = box_iou(t_boxes, p_boxes, xywh=False)
            tp = match(iou, t_cls, p_cls)

            self.stats["tp"].extend(tp)
            self.stats["conf"].extend(conf)
            self.stats["p_cls"].extend(p_cls)
            self.stats["t_cls"].extend(t_cls)

        return total_loss, loss_items

    def _on_epoch_start(self):
        super()._on_epoch_start()
        self.step = 0

        self.stats = {"tp": [],
                      "conf": [],
                      "p_cls": [],
                      "t_cls": []}

        self.avg_loss_items = {}
        self.val_loss_items = {}
        self.val_metrics = {}

        self.logger.update(**{"mAP50": "", "mAP50:95": ""})

    def _on_epoch_end(self):
        super()._on_epoch_end()

    def _on_iteration_end(self, loss_items, batch_labels):
        for k, v in loss_items.items():
            if k in self.avg_loss_items:
                self.avg_loss_items[k] = (self.avg_loss_items[k] * self.step + v) / (self.step + 1)
            else:
                self.avg_loss_items[k] = v

        self.env.update_info()

        self.images += len(batch_labels)
        self.instances += np.sum(np.sum(batch_labels[..., 1:5], -1) > 0)

        log = {"Ins/Img": f"{self.instances}/{self.images}",
               **self.avg_loss_items,
               **self.env.get_info()}

        if self.pbar.current == self.pbar.total - 1:
            result = ap_per_class(**self.stats)
            self.metric.update(result)

            precision = float(np.mean(self.metric.p)) if self.metric.p is not None else 0.0
            recall = float(np.mean(self.metric.r)) if self.metric.r is not None else 0.0
            map50 = float(self.metric.map50)
            map_val = float(self.metric.map)

            log["mAP50"] = map50
            log["mAP50:95"] = map_val

            self.val_loss_items = {k: float(v) for k, v in self.avg_loss_items.items()}
            self.val_metrics = {"precision": precision,
                                "recall": recall,
                                "mAP50": map50,
                                "mAP": map_val}

        self.logger.update(**log)
        self.pbar.set_status(**self.logger.data)

        self.step += 1

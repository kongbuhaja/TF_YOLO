from src.engine.handler import Handler
from src.utils.util import NMS
import numpy as np
from tqdm import tqdm


class Evaluator(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "eval")

    def evaluate(self):
        self._is_exist_dir()
        pred_json_path = self.eval_path / f"predictions.json"

        if not pred_json_path.exists():
            self._evaluate_step(pred_json_path)

        self._compute_metrics(pred_json_path)

    def _evaluate_step(self, pred_json_path):
        preds_json_list = []
        is_new_gt = not self.dataset.is_exist_gt()

        for data in tqdm(self.dataloader,
                         total=len(self.dataloader),
                         desc="Evaluating"):
            batch_image = data["image"]

            preds, _ = self.model(batch_image.astype(np.float32), training=False)

            nms_preds = NMS(preds,
                            conf_th=self.cfg.conf_th,
                            iou_th=self.cfg.iou_th,
                            max_det=self.cfg.max_det,
                            nc=self.model.nc)

            for b, (image_id, image, pred, info) in enumerate(zip(data["image_id"], data["image"], nms_preds, data["info"])):
                pred = pred.numpy()
                if len(pred) == 0:
                    continue

                rx, ry, l, t = info
                pad, ratio = np.array([l, t], pred.dtype), np.array([rx, ry], pred.dtpye)

                for cls_id, box, conf in zip(
                    pred[:, 5].astype(int), pred[:, :4], pred[:, 4]
                ):
                    xy = ((box[:2] - box[2:4] / 2) - pad) / ratio
                    wh = box[2:4] / ratio
                    box_original = np.concatenate([xy, wh])

                    pred_json = {
                        "image_id": int(image_id),
                        "category_id": self.category_map[cls_id]["id"],
                        "bbox": box_original.tolist(),
                        "score": float(conf),
                    }
                    preds_json_list.append(pred_json)

                if is_new_gt:
                    labels = data["labels"][b]
                    labels = labels[np.sum(labels[..., 1:5], -1) > 0]
                    self.dataset.add_gt(int(image_id), image, labels)

        if is_new_gt:
            self.dataset.save_gts()

        self.dataset.util.save_data(str(pred_json_path), preds_json_list)

    def _compute_metrics(self, pred_json_path):
        self.dataset.eval_metric(pred_json_path)

    def _is_exist_dir(self):
        self.eval_path = self.model.path / "evaluation"
        self.eval_path.mkdir(parents=True, exist_ok=True)

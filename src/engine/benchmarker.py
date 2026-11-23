import time
import yaml
import numpy as np
from pathlib import Path

from src.engine.handler import Handler
from src.tools.util import NMS, artifact_size_mb


class Benchmarker(Handler):
    def __init__(self, env, cfg, dataset):
        cfg = cfg.copy()
        cfg.batch_size = 1
        super().__init__(env, None, cfg, dataset, "eval")

    def run(self, model, source_model=None):
        device_name = model.device_name
        format, dtype = model.format, model.dtype
        artifact_path = model.artifact_path
        try:
            self._warmup(model)

            latencies, pred_json_path = self._inference_loop(model)
            metrics = self._compute_metrics(pred_json_path)

            params_M, gflops = self._model_stats(model, source_model)

            latencies = np.array(latencies)
            model_size = artifact_size_mb(artifact_path)

            results = {
                "dataset": self.dataset.name,
                "format": format,
                "dtype": dtype,
                "pred_dtype": getattr(model, "output_dtype", "unknown"),
                "device": device_name,
                "device_id": model.gpu if not isinstance(model.gpu, str) else model.gpu,
                "mAP": metrics.get("mAP", 0.0),
                "mAP50": metrics.get("mAP50", 0.0),
                "mAP75": metrics.get("mAP75", 0.0),
                "latency_ms": float(np.mean(latencies) * 1000),
                "latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
                "fps": float(1.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0,
                "params_M": params_M,
                "GFLOPs": gflops,
                "model_size_MB": model_size,
            }
            self._print_results(results)
            self._write_summary(model, results)
            return results
        except Exception as e:
            print(f"[Benchmark] {format}/{dtype} on {device_name} failed: {e}")
            return None

    def _model_stats(self, model, source_model):
        if source_model is not None:
            from src.tools.exporter import count_params, count_flops
            return (count_params(source_model) / 1e6,
                    count_flops(source_model, self.cfg.input_shape) / 1e9)
        export_yaml = Path(model.artifact_path).parent / "export.yaml"
        if export_yaml.exists():
            with open(export_yaml) as f:
                summary = yaml.safe_load(f) or {}
            return summary.get("params_M", 0.0), summary.get("GFLOPs", 0.0)
        return 0.0, 0.0

    def _warmup(self, model, n=100):
        dummy = np.zeros(self.cfg.input_shape, dtype=np.uint8)[None]
        for _ in range(n):
            model.infer(dummy)

    def _inference_loop(self, model):
        preds_json_list = []
        is_new_gt = not self.dataset.is_exist_gt()
        latencies = []

        total_images = len(self.dataloader)
        total_dets = 0

        for i, data in enumerate(self.pbar):
            batch_image = data["image"]

            t0 = time.perf_counter()
            preds = model.infer(batch_image)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            preds = model.to_float32(preds)

            nms_preds = NMS(preds,
                            conf_th=self.cfg.conf_th,
                            iou_th=self.cfg.iou_th,
                            max_det=self.cfg.max_det,
                            nc=model.nc)

            for b, (image_id, image, pred, info) in enumerate(
                    zip(data["image_id"], data["image"], nms_preds, data["info"])):
                pred = np.asarray(pred)
                if len(pred) == 0:
                    continue

                rx, ry, l, t = info
                pad = np.array([l, t], pred.dtype)
                ratio = np.array([rx, ry], pred.dtype)
                bboxes = np.concatenate(
                    [(pred[:, :2] - pad) / ratio,
                     (pred[:, 2:4] - pred[:, :2]) / ratio], axis=-1)
                confs, class_ids = pred[:, 4], pred[:, 5]

                pred_json = [{"image_id": int(image_id),
                              "category_id": self.category_map[int(cls_id)]["id"],
                              "bbox": bbox.tolist(),
                              "score": float(conf)}
                             for bbox, conf, cls_id in zip(bboxes, confs, class_ids)]
                preds_json_list.extend(pred_json)
                total_dets += len(pred_json)

                if is_new_gt:
                    labels = data["labels"][b]
                    labels = labels[np.sum(labels[..., 1:5], -1) > 0]
                    self.dataset.add_gt(int(image_id), image, labels)

            self.env.update_info()
            self.monitor.update(images=f"{i + 1}/{total_images}",
                                dets=total_dets,
                                **self.env.get_info())
            self.pbar.set_status(**self.monitor.data)

        if is_new_gt:
            self.dataset.save_gts()

        out_dir = self._bench_path(model) / f"{model.format}_{model.dtype}"
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_json_path = out_dir / f"pred_{model.format}_{model.dtype}.json"
        self.dataset.util.save_data(str(pred_json_path), preds_json_list)
        return latencies, pred_json_path

    def _compute_metrics(self, pred_json_path):
        gt_json_path = self.dataset.dirs["eval"]
        if not Path(gt_json_path).exists():
            print(f"GT annotations not found: {gt_json_path}")
            return {}

        if not pred_json_path.exists() or pred_json_path.stat().st_size <= 2:
            print("[Benchmark] No usable predictions; mAP = 0")
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

        stats = self.dataset.eval_metric(pred_json_path)
        return {
            "mAP": float(stats[0]),
            "mAP50": float(stats[1]),
            "mAP75": float(stats[2]),
        }

    def _bench_path(self, model):
        bench_path = model.path / "benchmark"
        bench_path.mkdir(parents=True, exist_ok=True)
        return bench_path

    def _print_results(self, r):
        line = "=" * 64
        print()
        print(line)
        print(" Benchmark Results")
        print(line)
        print(f" Format:     {r['format']:<12} Dtype: {r['dtype']:<8} Pred: {r['pred_dtype']}")
        print(f" Device:     {r['device']}")
        print("-" * 64)
        print(f" mAP:        {r['mAP']:.4f}       mAP50: {r['mAP50']:.4f}   mAP75: {r['mAP75']:.4f}")
        print(f" Latency:    {r['latency_ms']:.2f} ms     P95: {r['latency_p95_ms']:.2f} ms    FPS: {r['fps']:.1f}")
        print(f" Params:     {r['params_M']:.2f} M       GFLOPs: {r['GFLOPs']:.2f}")
        print(f" Model Size: {r['model_size_MB']:.2f} MB")
        print(line)

    def _write_summary(self, model, results):
        out_dirs = [Path(model.artifact_path).parent,
                    self._bench_path(model) / f"{results['format']}_{results['dtype']}"]
        for out_dir in out_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / "benchmark.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)
            print(f"[Benchmark] summary -> {path}")

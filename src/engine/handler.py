from src.data import Dataloader
from src.utils.logger import Logger
from pathlib import Path
from src.utils.progress import ProgressBar
import os, yaml, shutil


class _FlowListDumper(yaml.SafeDumper):
    pass


def _flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def _block_dict(dumper, data):
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=False)


_FlowListDumper.add_representer(list, _flow_list)
_FlowListDumper.add_representer(tuple, _flow_list)
_FlowListDumper.add_representer(dict, _block_dict)


class Handler():
    def __init__(self, env, model, cfg, dataset, split):
        self.env = env
        self.model = model
        self.cfg = cfg
        self.dataset = dataset
        self.split = split
        self.logger = Logger(self.split)
        data = self.dataset.load("val", self.cfg.cache, self.cfg.workers)

        self.dataloader = Dataloader(cfg, data)
        self.category_map = data["category_map"]
        self.pbar = ProgressBar(self.dataloader,
                                task=self.cfg.task.upper(),
                                split=self.split,
                                headers=self.logger.keys)

    def make_dir(self, name):
        path = Path.cwd().resolve() / "results"
        os.makedirs(path, exist_ok=True)

        dir_name = getattr(self.cfg, "name", None)
        if not dir_name:
            n = sum([d.startswith(name) for d in os.listdir(path)])
            dir_name = f"{name}{n}"

        self.cfg.path = path / dir_name
        os.makedirs(self.cfg.path, exist_ok=True)
        self.cfg.weights = self.cfg.path / "weights"
        os.makedirs(self.cfg.weights, exist_ok=True)

        self._save_args()

    def _save_args(self):
        args = {}
        skip_keys = {"path", "weights", "environment", "model"}
        for key, value in self.cfg.items():
            if key in skip_keys:
                continue
            if key == "data":
                args[key] = self._serialize(value, order=["name", "path", "file", "dirs", "urls", "classes"])
            else:
                args[key] = self._serialize(value)

        keys = list(args.keys())
        if "name" in keys:
            keys.remove("name")
            keys.insert(0, "name")
        if "data" in keys:
            keys.remove("data")
            keys.append("data")
        args = {k: args[k] for k in keys}

        args_path = self.cfg.path / "args.yaml"
        with open(args_path, "w") as f:
            yaml.dump(args, f, Dumper=_FlowListDumper, default_flow_style=None, sort_keys=False)

    def _serialize(self, value, order=None):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "items"):
            d = {k: self._serialize(v) for k, v in value.items()}
            if order:
                return {k: d[k] for k in order if k in d}
            return d
        if isinstance(value, (list, tuple)):
            return [self._serialize(v) for v in value]
        try:
            yaml.safe_dump(value)
            return value
        except:
            return str(value)

    def on_epoch_start(self):
        self.images = 0
        self.instances = 0

    def on_epoch_end(self):
        pass

    def save_model(self, epoch, best_map, current_map):
        epoch_dir = self.cfg.weights / f"epoch{epoch}"
        last_dir = self.cfg.weights / "last"
        best_dir = self.cfg.weights / "best"

        for d in [epoch_dir, last_dir]:
            if d.exists():
                shutil.rmtree(str(d))
            self.model.save(str(d))

        if current_map > best_map:
            if best_dir.exists():
                shutil.rmtree(str(best_dir))
            self.model.save(str(best_dir))
            return True
        return False

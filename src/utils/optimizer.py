import tensorflow as tf
from src.utils.scheduler import Scheduler

class Optimizer():
    def __init__(self, cfg, total_steps, steps_per_epoch):
        opt_cfg = getattr(cfg, "optimizer", {"name": "SGD",
                                             "beta": 0.9,
                                             "weight_decay": 1e-5,
                                             "warmup_beta": 0.8})
        sch_cfg = getattr(cfg, "scheduler", {"name": "linear",
                                             "lri": 0.01,
                                             "lrf": 0.01,
                                             "warmup_epochs": 3,
                                             "warmup_bias_lr": 0.1})
        self.scheduler = Scheduler(sch_cfg, total_steps, steps_per_epoch)

        self.warmup_steps = self.scheduler.warmup_steps

        for key, value in opt_cfg.items():
            if key != "name":
                value = float(value)
            setattr(self, key, value)
        
        self.build_optimizer(self.name.lower())

    def build_optimizer(self, name):
        opt_dict = {
            "sgd": tf.keras.optimizers.SGD,
            "adam": tf.keras.optimizers.Adam,
            "adamw": tf.keras.optimizers.AdamW,
            "adamax": tf.keras.optimizers.Adamax,
            "nadam": tf.keras.optimizers.Nadam,
            "rmsprop": tf.keras.optimizers.RMSprop
        }

        try:
            model = opt_dict[self.name.lower()]
        except:
            print(f"Optimizer '{self.name} not found. Defaulting to 'SGD'.")
            self.name = "SGD"
            name = self.name.lower()
            model = opt_dict[name]

        args = {"learning_rate": self.scheduler}
        
        if name == "sgd":
            args["momentum"] = self.beta
        elif name in ["adam", "adamw"]:
            args["beta_1"] = self.beta
            args["weight_decay"] = self.weight_decay
        elif name in ["adamax", "nadam"]:
            args["beta_1"] = self.beta
        elif name in ["rmsprop"]:
            args["rho"] = self.beta
            args["weight_decay"] = self.weight_decay
            
        self.model = model(**args)

    def update(self, step):
        step = tf.cast(step, tf.float32)

        if step < self.warmup_steps:
            progress = step / tf.maximum(self.warmup_steps, 1.0)
            beta = self.warmup_beta + (self.beta - self.warmup_beta) * progress

            if hasattr(self.model, "momentum"):
                if isinstance(self.model.momentum, tf.Variable):
                    self.model.momentum.assign(beta)
                else:
                    self.model.momentum = beta

            elif hasattr(self.model, "beta_1"):
                if isinstance(self.model.beta_1, tf.Variable):
                    self.model.beta_1.assign(beta)
                else:
                    self.model.beta_1 = beta

            elif hasattr(self.model, "rho"):
                if isinstance(self.model.rho, tf.Variable):
                    self.model.rho.assign(beta)
                else:
                    self.model.rho = beta

    @property
    def lr(self):
        # return self.model.learning_rate(self.model.iterations)
        return self.model.learning_rate
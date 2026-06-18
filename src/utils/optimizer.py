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

        amp = str(getattr(cfg, "amp", False))
        self.use_loss_scaling = (amp == "fp16")

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
            opt = opt_dict[self.name.lower()]
        except:
            print(f"Optimizer '{self.name}' not found. Defaulting to 'SGD'.")
            self.name = "SGD"
            name = self.name.lower()
            opt = opt_dict[name]

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
            
        self.model = opt(**args)
        if self.use_loss_scaling:
            self.model = tf.keras.mixed_precision.LossScaleOptimizer(self.model)

    def get_scaled_loss(self, loss):
        if self.use_loss_scaling:
            return self.model.get_scaled_loss(loss)
        return loss

    def get_unscaled_gradients(self, grads):
        if self.use_loss_scaling:
            return self.model.get_unscaled_gradients(grads)
        return grads

    def update(self, step):
        step = tf.cast(step, tf.float32)

        if step < self.warmup_steps:
            progress = step / tf.maximum(self.warmup_steps, 1.0)
            beta = self.warmup_beta + (self.beta - self.warmup_beta) * progress

            opt = getattr(self.model, "inner_optimizer", self.model)

            if hasattr(opt, "momentum"):
                if isinstance(opt.momentum, tf.Variable):
                    opt.momentum.assign(beta)
                else:
                    opt.momentum = beta

            elif hasattr(opt, "beta_1"):
                if isinstance(opt.beta_1, tf.Variable):
                    opt.beta_1.assign(beta)
                else:
                    opt.beta_1 = beta

            elif hasattr(opt, "rho"):
                if isinstance(opt.rho, tf.Variable):
                    opt.rho.assign(beta)
                else:
                    opt.rho = beta

    @property
    def lr(self):
        opt = getattr(self.model, "inner_optimizer", self.model)
        return opt.learning_rate
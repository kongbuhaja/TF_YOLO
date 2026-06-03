import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import math

class Scheduler(LearningRateSchedule):
    def __init__(self, cfg, total_steps, steps_per_epoch):
        super().__init__()
        for key, value in cfg.items():
            if key != "name":
                value = float(value)
            setattr(self, key, value)
        self.total_steps = total_steps
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        
        self.build_scheduler(self.name.lower())

    def build_scheduler(self, name):
        sch_dict = {
            "linear": self.linear_decay,
            "cosine": self.cosine_decay,
            "constant": self.constant
        }

        try:
            self.model = sch_dict[name]
        except:
            print(f"Scheduler '{self.name}' not found. Defaulting to 'linear'.")
            self.model = sch_dict["linear"]

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        self.lr = tf.cond(step < self.warmup_steps,
                         lambda: self.warmup_model(step),
                         lambda: self.model(step))

        return self.lr
    
    def warmup_model(self, step):
        progress = step / tf.maximum(self.warmup_steps, 1.0)
        return self.warmup_bias_lr + (self.lri - self.warmup_bias_lr) * progress
    
    def warmup_progress(self, step):
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return tf.minimum(progress, 1.0)
    
    def constant(self, step):
        return self.lri

    def linear_decay(self, step):
        # end_lr = self.lri * self.lrf
        # return self.lri + progress * (end_lr - self.lri)
        progress = self.warmup_progress(step)
        decay = (1.0 - progress) * 1.0 + progress * self.lrf
        return self.lri * decay        
    
    def cosine_decay(self, step):
        progress = self.warmup_progress(step)
        cosine_value = 0.5 * (1.0 + tf.math.cos(math.pi * progress))
        decay = (1.0 - self.lrf) * cosine_value + self.lrf
        return self.lri * decay
    
    def get_config(self):
        return {
            "name": self.name,
            "lri": self.lri,
            "lrf": self.lrf,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "warmup_bias_lr": self.warmup_bias_lr
        }

from src.engine.handler import Handler
from src.engine.validator import Validator
from src.utils.loss import DFLDetectionLoss
from src.utils.optimizer import Optimizer
import tensorflow as tf
import numpy as np

class Trainer(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "train")
        # super().__init__(env, model, cfg, dataset, "val")
        self.loss = DFLDetectionLoss(model, cfg.loss)
        self.steps_per_epoch = len(self.dataloader)
        self.global_step = 0
        self.total_steps = self.cfg.epochs * self.steps_per_epoch
        self.optimizer = Optimizer(cfg, self.total_steps, self.steps_per_epoch)

        self.validator = Validator(self.env, self.model, self.cfg, self.dataset) if self.cfg.period > 0 else None


        self.make_dir(self.model.model_name)
        
    def train(self):
        try:
            for epoch in range(self.cfg.epochs):
                self.on_epoch_start()

                for data in self.pbar:
                    batch_image, batch_labels = data["image"], data["labels"]

                    self.on_iteration_start()

                    total_loss, loss_items = self.train_step(batch_image, batch_labels)

                    self.on_iteration_end(epoch, loss_items, batch_labels)

                self.on_epoch_end(epoch)

        except Exception as e:
            total_loss, loss_items = 0.0, {}
            print(f"Training loop iterrupted: {e}")
            raise e

        return total_loss, loss_items

    @tf.function
    def train_step(self, batch_image, batch_labels):
        with tf.GradientTape() as tape:
            raw_preds = self.model(batch_image, training=True)
            total_loss, loss_items = self.loss(raw_preds, batch_labels)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.model.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, loss_items
    
    def on_epoch_start(self):
        super().on_epoch_start()

    def on_epoch_end(self, epoch):
        super().on_epoch_end()
        if self.validator and epoch % self.cfg.period == 0:
            self.validator.validate()

    def on_iteration_start(self):
        self.optimizer.update(self.global_step)

    def on_iteration_end(self, epoch, loss_items, batch_labels):
        def log_update():
            self.images += len(batch_labels)
            self.instances += np.sum(batch_labels[..., 0] != -1)

            log = {"Epoch": f"{epoch+1}/{self.cfg.epochs}",
                   "Ins/Img": f"{self.instances}/{self.images}",
                   "LR": self.optimizer.lr,
                   **loss_items,
                   **self.env.get_info()}

            self.logger.update(**log)

        self.env.update_info()
        log_update()
        self.pbar.set_status(**self.logger.data)

        self.global_step += 1
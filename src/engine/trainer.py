from src.engine.handler import Handler
from src.utils.loss import DFLDetectionLoss
from src.utils.optimizer import Optimizer
from src.utils.progress import ProgressBar
import tensorflow as tf

class Trainer(Handler):
    def __init__(self, env, model, cfg, dataset):
        super().__init__(env, model, cfg, dataset, "train")
        self.loss = DFLDetectionLoss(model, cfg.loss)
        self.steps_per_epoch = len(self.dataloader)
        self.epochs = self.cfg.epochs
        self.total_steps = self.epochs * self.steps_per_epoch
        self.optimizer = Optimizer(cfg, self.total_steps, self.steps_per_epoch)

        self.make_dir(self.model.model_name)
        
    def train(self):
        def log_update():
            self.env.update_info()
            self.logger.update(Epoch=f"{epoch+1}/{self.epochs}")
            self.logger.update(**loss_items)
            self.logger.update(Lr=self.optimizer.lr)
            self.logger.update(**self.env.get_info_log())

        try:            
            for epoch in range(self.epochs):
                pbar = ProgressBar(self.dataloader,
                                   task=self.cfg.task,
                                   split=self.split,
                                   headers=self.logger.keys)
                for data in pbar:
                    batch_image, batch_labels = data["image"], data["labels"]
                    
                    total_loss, loss_items = self.train_step(batch_image, batch_labels)

                    log_update()
                    pbar.set_status(**self.logger.data)

        except Exception as e:
            total_loss, loss_items = 0.0, {}
            print(f"Training loop iterrupted: {e}")
            raise e
        
        return total_loss, loss_items

    @tf.function
    def train_step(self, batch_image, batch_labels):
        with tf.GradientTape() as tape:
            preds = self.model(batch_image, training=True)
            total_loss, loss_items = self.loss(preds, batch_labels)
            
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.model.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return total_loss, loss_items
    
    def on_epoch_start(self):
        pass

        
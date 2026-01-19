from src.engine.handler import Handler
from src.utils.loss import DFLDetectionloss
from tqdm import tqdm
import tensorflow as tf

class Trainer(Handler):
    def __init__(self, model, cfg, dataset):
        super().__init__(model, cfg, dataset, "train")
        self.loss = DFLDetectionloss(model)
        self.make_dir(self.model.name)

    def __call__(self, model=None):
        if model is None:
            model = self.model
        else:
            self.loss = DFLDetectionloss(model)
        i = 0
        try:
            for data in tqdm(self.dataloader,
                            total=len(self.dataloader),
                            desc=f"Train {self.dataset.name} data"):
                batch_image, batch_labels = data["image"], data["labels"]
                return batch_image, self.train_step(batch_image, batch_labels)
                
                # total_loss, loss_items = self.train_step(batch_image, batch_labels)
                # return total_loss, loss_items

        finally:
            self.dataloader.on_epoch_end()

    # @tf.function
    def train_step(self, images, gts):
        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            # total_loss, loss_items = self.loss(preds, gts)
            result = self.loss(preds, gts)
        return result
        # gradients = tape.gradient(total_loss, self.model.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variable))
        
        # return total_loss, loss_items

        
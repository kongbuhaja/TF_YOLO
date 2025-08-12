from src.data.dataset import Dataset
import numpy as np
from src.data.process import Process, Augmentation
from threading import Thread
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy
import logging

class Dataloader():
    def __init__(self, cfg, data=None):
        for name, value in cfg.__dict__.items():
            setattr(self, name, value)
        self.epoch = 0
        self.close_mosaic = self.epochs - self.close_mosaic + 1

        self._executor = None
        self._prefetch_thread = None
        self._batch_queue = None


        if data:
            self.insert_data(data)

    def insert_data(self, data):
        self.dtype = data["dtype"]
        self.data = data["data"]
        self.indices = np.arange(len(self.data))

        # process
        preprocess = Process({"read_image": True})

        if self.dtype == "train" or self.dtype == "val":
            process = Process(self.augmentation) + Process(self.process)
        elif self.dtype == "val":
            process = Process(self.process)
        else:
            self.batch_size = 1
            process = Process({"resize_padding_with_info": self.process["resize_padding"][:2],
                               "unsqueeze_coords": self.process["unsqueeze_coords"],
                               "segment_to_task": self.process["segment_to_task"]})
            
        self.process_pipeline = preprocess + process
        self.postprocess_pipeline = Process(self.postprocess)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __iter__(self):
        self.on_epoch_start()
        return self
    
    def __next__(self):
        if self.idx >= len(self):
            self.on_epoch_end()
            raise StopIteration
        
        if self._batch_queue:
            batch_data = self._batch_queue.get()
        else:
            start_idx = self.idx * self.batch_size
            end_idx = min((self.idx + 1) * self.batch_size, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx].tolist()
            batch_data = self._process_batch(batch_indices)

        if isinstance(batch_data, Exception):
            logging.error(f"Batch error, fallback: {batch_data}")
            raise StopIteration
        self.idx += 1

        return batch_data
    
    def _prefetch_worker(self):
        while not self._stop_signal and self.prefetch_idx < len(self):
            try:
                start_idx = self.prefetch_idx * self.batch_size
                end_idx = min((self.prefetch_idx + 1) * self.batch_size, len(self.indices))
                batch_indices = self.indices[start_idx:end_idx].tolist()
                batch_data = self._process_batch(batch_indices)
                
                if not self._stop_signal:
                    self._batch_queue.put(batch_data, timeout=5)
                    self.prefetch_idx += 1

            except Exception as e:
                logging.error(f"Prefetch worker error: {e}")
                self._batch_queue.put(e)
                break

    def _process_batch(self, batch_indices):
        try:            
            if getattr(self.process_pipeline, "use_mosaic", False):
                sample_indices = [[idx] + np.random.choice(self.indices, 3).tolist() for idx in batch_indices]
                batch_data = [[self.data[idx].copy() for idx in indices] for indices in sample_indices]
            else:
                batch_data = [self.data[idx].copy() for idx in batch_indices]
            
            if self.workers > 1 and len(batch_data) > 1:
                futures = [self._executor.submit(self.process_pipeline, data) for data in batch_data]
                processed_data = [future.result() for future in futures]
            else:
                processed_data = [self.process_pipeline(data) for data in batch_data]
            
            final_data = self.postprocess_pipeline(processed_data)

            return final_data
        
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return e
    
    def _choose_executor(self):
        use_processes = getattr(self.process_pipeline, "use_mosaic", False) and self.workers > 2
        if use_processes:
            return ProcessPoolExecutor(max_workers=self.workers)
        return ThreadPoolExecutor(max_workers=self.workers)

    def on_epoch_start(self):
        self.idx = 0
        self.prefetch_idx = 0
        self.epoch += 1
        
        if self.dtype == "train" or self.dtype == "val":
            if self.shuffle:
                np.random.shuffle(self.indices)
            if self.epoch == self.close_mosaic:
                self.process_pipeline.close_mosaic()

        self._stop_signal = False

        if self.workers > 1:
            self._executor = self._choose_executor()

        if self.prefetch_size > 0:
            self._batch_queue = Queue(maxsize=self.prefetch_size)
            self._prefetch_thread = Thread(target=self._prefetch_worker, daemon=True)
            self._prefetch_thread.start()
    
    def on_epoch_end(self):
        self._stop_signal = True

        while self._batch_queue and not self._batch_queue.empty():
            try:
                self._batch_queue.get_nowait()
            except Empty:
                break

        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3)
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


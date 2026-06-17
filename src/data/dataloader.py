from src.data.dataset import Dataset
import numpy as np
from src.data.process import Process
from threading import Thread
from queue import Queue, Empty, Full
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from copy import deepcopy
import logging, traceback

class Dataloader():
    def __init__(self, cfg, data):
        self.epoch = 0
        self.close_mosaic = cfg.epochs - cfg.close_mosaic + 1
        self.batch_size = cfg.batch_size
        self.prefetch_size = cfg.prefetch_size
        self.shuffle = cfg.shuffle
        self.workers = cfg.workers
        
        self._executor = None
        self._prefetch_thread = None
        self._batch_queue = None

        self.insert_data(cfg.copy(), data)

    def insert_data(self, cfg, data):
        self.split = data["split"]
        self.data = data["data"]
        self.indices = np.arange(len(self.data))

        # process
        preprocess = Process({"read_image": True})
        
        process = Process({"resize_padding": [cfg.input_shape, cfg.constant]})
        if self.split == "train":
            process = Process(cfg.augmentation) + process
        elif self.split == "eval":
            process = Process({"resize_padding_with_info": [cfg.input_shape, cfg.constant]})
        process += Process(cfg.process)

        if self.split != "train":
            process.filter.ratio = 0 
            
        self.process_pipeline = preprocess + process
        self.postprocess_pipeline = Process(cfg.postprocess)

    def __len__(self):
        if self.split == "train":
            return len(self.indices) // self.batch_size
        else:
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
            batch_indices = self.indices[start_idx:end_idx]
            batch_data = self._process_batch(batch_indices)

        if isinstance(batch_data, Exception):
            logging.error(f"Batch error occurred! Stopping iteration.")
            self.on_epoch_end()
            raise StopIteration
        
        self.idx += 1

        return batch_data
    
    def _prefetch_worker(self):
        while not self._stop_signal and self.prefetch_idx < len(self):
            try:
                start_idx = self.prefetch_idx * self.batch_size
                end_idx = min((self.prefetch_idx + 1) * self.batch_size, len(self.indices))
                batch_indices = self.indices[start_idx:end_idx]
                batch_data = self._process_batch(batch_indices)
                
                while not self._stop_signal:
                    try:
                        self._batch_queue.put(batch_data, timeout=5)
                        self.prefetch_idx += 1
                        break
                    except Full:
                        continue

            except Exception as e:
                logging.error(f"Prefetch worker error:\n{traceback.format_exc()}")
                try:
                    self._batch_queue.put(e, timeout=5)
                except:
                    pass
                break

    def _process_batch(self, batch_indices):
        try:            
            if hasattr(self.process_pipeline, "mosaic"):
                sample_indices = [[idx] + np.random.choice(self.indices, 3).tolist() for idx in batch_indices]
                batch_data = [[self.data[idx].copy() for idx in indices] for indices in sample_indices]
            else:
                batch_data = [self.data[idx].copy() for idx in batch_indices]

            if self.workers > 1 and len(batch_data) > 1:
                futures = [self._executor.submit(self.process_pipeline, data) for data in batch_data]
                processed_data = [future.result() for future in futures]
            else:
                processed_data = [self.process_pipeline(data) for data in batch_data]

            batch_data = self.postprocess_pipeline(processed_data)
            return batch_data
        
        except Exception as e:
            logging.error(f"Error processing batch: \n{traceback.format_exc()}")
            return e
    
    def _choose_executor(self):
        use_processes = hasattr(self.process_pipeline, "mosaic") and self.workers > 2
        if use_processes:
            return ProcessPoolExecutor(max_workers=self.workers)
        return ThreadPoolExecutor(max_workers=self.workers)

    def on_epoch_start(self):
        self.idx = 0
        self.prefetch_idx = 0
        self.epoch += 1
        
        if self.split == "train":
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
